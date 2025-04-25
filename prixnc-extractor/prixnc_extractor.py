# Native imports
from typing import List, Dict, Any, Optional
import random
import logging

# API client imports
import requests
from requests import Response
from requests.exceptions import RequestException, Timeout, ConnectionError

from tenacity import (
    Retrying,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    RetryError
)

# Export imports
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4, landscape
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph
from reportlab.lib.styles import getSampleStyleSheet


# Logger configuration with timestamp and severity level
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class SessionManager:
    """Manages a pool of HTTP sessions for connection reuse and performance.
    
    Attributes:
        sessions: List of active HTTP sessions.
        
    Note:
        - Reusing sessions avoids repeated TLS handshake overhead.
        - Explicit session closing prevents resource leaks.
    """

    sessions: List[requests.Session]

    def __init__(self, nb_session: int = 3) -> None:
        """Initializes the session pool.
        
        Args:
            nb_session: Number of sessions to create (default: 3).
        """
        self.sessions = [requests.Session() for _ in range(nb_session)]

    def _get_session(self) -> requests.Session:
        """Returns a random session from the pool.
        
        Returns:
            A reusable HTTP session.
            
        Note:
            Random selection provides basic load balancing.
        """
        return random.choice(self.sessions)

    def _close_sessions(self) -> None:
        """Gracefully closes all HTTP sessions.
        
        Important:
            Must be called before object destruction to avoid
            socket leaks.
        """
        for session in self.sessions:
            session.close()


class APIClient(SessionManager):
    """Robust API client with error handling and automatic retries.
    
    Features:
        - Exponential backoff for network errors
        - Configurable timeouts
        - Centralized session management
        
    Warning:
        Not suitable for non-idempotent requests (POST/PUT)
        without disabling retries.
    """

    _api: str
    _timeout: int
    _max_retry: int
    _min_retry_time: int
    _max_retry_time: int
    _retry_time_multiplier: int

    def __init__(
        self,
        timeout: int = 5,
        max_retry: int = 3,
        min_retry_time: int = 2,
        max_retry_time: int = 10,
        retry_time_multiplier: int = 1
    ) -> None:
        """Configures HTTP request strategy.
        
        Args:
            timeout: Maximum wait time (seconds).
            max_retry: Maximum retry attempts.
            min_retry_time: Minimum retry delay (seconds).
            max_retry_time: Maximum retry delay (seconds).
            retry_time_multiplier: Exponential backoff factor.
            
        Raises:
            ValueError: If timeout <= 0.
        """
        if timeout <= 0:
            raise ValueError("Timeout must be positive")
            
        super().__init__()
        self._timeout = timeout
        self._max_retry = max_retry
        self._min_retry_time = min_retry_time
        self._max_retry_time = max_retry_time
        self._retry_time_multiplier = retry_time_multiplier

    def _api_call(
        self,
        api_url: str,
        params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any] | None:
        """Executes an API request with automatic error handling.
        
        Args:
            api_url: Full endpoint URL.
            params: Optional query parameters.
            
        Returns:
            Parsed JSON data or None on failure.
            
        Raises:
            RetryError: When all attempts fail.
            ValueError: If response contains invalid JSON.
            
        Note:
            4XX errors (except 429) don't trigger retries.
        """
        try:
            retry_configuration = Retrying(
                stop=stop_after_attempt(self._max_retry),
                wait=wait_exponential(
                    multiplier=self._retry_time_multiplier,
                    min=self._min_retry_time,
                    max=self._max_retry_time
                ),
                retry=retry_if_exception_type(
                    (RequestException, ConnectionError, TimeoutError)
                )
            )
            
            for attempt in retry_configuration:
                with attempt:
                    self._session = self._get_session()
                    logger.info(f"Attempt #{attempt.retry_state.attempt_number} at {api_url}")
                    response = self._session.get(
                        api_url,
                        timeout=self._timeout,
                        params=params
                    )
                    response.raise_for_status()
                    return response.json()
                    
        except requests.HTTPError as e:
            logger.error(f"HTTP Error {e.response.status_code}: {e}")
        except ValueError as e:
            logger.error(f"Invalid JSON response: {e}")
        except Exception as e:
            logger.error(f"Unexpected error: {e.__class__.__name__}: {e}")
            
        return None


class PrixNcExctractService(APIClient):
    """Specialized client for PrixNc API (https://prix.nc).
    
    Features:
        - Automatic pagination handling
        - Data cleaning
        - Metadata (_links) management
    """

    _nc_api: str = "https://prix.nc/api/v1/produits/"
    _init_page: int = 0
    _current_page_url: str

    def __init__(self, bash_size: int = 500) -> None:
        """Initializes the service with default page size.
        
        Args:
            bash_size: Items per page (max 1000 per API limits).
        """
        self._current_page_url = (
            f"{self._nc_api}?page={self._init_page}&size={bash_size}"
        )
        super().__init__()

    def load_data(self) -> List[Dict[str, Any]] | None:
        """Loads all available products via paginated API.
        
        Returns:
            Raw product list or None on critical failure.
            
        Note:
            This is a blocking operation that may take significant time
            for large collections.
        """
        products: List[Dict[str, Any]] = []
        
        while True:
            json_response = self._api_call(self._current_page_url)
            if not json_response:
                logger.error(f"Failed loading {self._current_page_url}")
                return None
                
            # Merge paginated results
            products.extend(json_response["_embedded"]["produits"])
            
            # Pagination handling
            if "next" in json_response["_links"]:
                self._current_page_url = json_response["_links"]["next"]["href"]
            else:
                logger.info("Loading complete (%d products)", len(products))
                self._close_sessions()
                break
                
        return products

    def cleaning_data(
        self,
        products: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Cleans raw API data.
        
        Args:
            products: Raw data to clean.
            
        Returns:
            Cleaned data without unnecessary metadata.
            
        Example:
            >>> raw_data = [{"_links": {...}, "id": 1}]
            >>> cleaned = cleaning_data(raw_data)
            >>> assert "_links" not in cleaned[0]
        """
        return [
            {k: v for k, v in product.items() if k != "_links"}
            for product in products
        ]


class JSONExporter:
    """Exports data to multiple formats (CSV, Excel, PDF)."""
    
    def __init__(self, data: List[Dict[str, Any]]) -> None:
        """Initializes exporter with structured data.
        
        Args:
            data: List of dictionaries to export.
        """
        self.data = data
        self.df = pd.DataFrame(data)

    def to_csv(self, filepath: str) -> None:
        """Exports to CSV.
        
        Args:
            filepath: Output file path.
        """
        self.df.to_csv(filepath, index=False)
        logger.info("CSV file created: %s", filepath)

    def to_excel(self, filepath: str) -> None:
        """Exports to Excel.
        
        Args:
            filepath: Output file path.
            
        Note:
            Requires `openpyxl` package.
        """
        self.df.to_excel(filepath, index=False, engine="openpyxl")
        logger.info("Excel file created: %s", filepath)

    def to_pdf(self, filepath: str, title: Optional[str] = None, font_size: int = 8) -> None:
        """Exports data to PDF with optimized layout for wide tables.
        
        Args:
            filepath: Output file path
            title: Optional report title
            font_size: Base font size (adjust for more/less content)
        """
        if not self.data:
            logger.warning("No data to export to PDF")
            return

        try:
            # Custom page size for wide tables (W x H in points)
            # A4 landscape: 841.89 x 595.26 pts
            # We'll extend width while keeping height standard
            CUSTOM_WIDTH = 1200  # pts (~42 cm)
            CUSTOM_HEIGHT = 595.26  # Standard A4 landscape height
            
            doc = SimpleDocTemplate(
                filepath,
                pagesize=(CUSTOM_WIDTH, CUSTOM_HEIGHT),
                rightMargin=20,
                leftMargin=20,
                topMargin=30,
                bottomMargin=30
            )
            
            elements = []
            styles = getSampleStyleSheet()
            
            if title:
                # Use smaller title font for wide tables
                title_style = styles['Title']
                title_style.fontSize = font_size + 2
                elements.append(Paragraph(title, title_style))
            
            # Prepare table data
            table_data = [list(self.df.columns)] + self.df.values.tolist()
            
            # Calculate dynamic column widths
            def calculate_widths():
                # Base width per character (adjust based on font)
                CHAR_WIDTH = font_size * 0.6  
                MIN_WIDTH = 40  # pts
                MAX_WIDTH = 200  # pts
                
                widths = []
                for col_idx in range(len(table_data[0])):
                    # Get max content length in column
                    max_len = max(
                        len(str(row[col_idx])) 
                        for row in table_data
                    )
                    # Calculate proposed width
                    col_width = min(
                        max(CHAR_WIDTH * max_len, MIN_WIDTH),
                        MAX_WIDTH
                    )
                    widths.append(col_width)
                return widths
            
            col_widths = calculate_widths()
            
            # Adjust row height based on font size
            row_height = max(font_size + 4, 16)
            
            # Create table
            table = Table(
                table_data,
                colWidths=col_widths,
                rowHeights=row_height,
                repeatRows=1
            )
            
            # Table styling
            table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#4F81BD')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),  # Left align for better readability
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, -1), font_size),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 6),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
                ('VALIGN', (0, 0), (-1, -1), 'TOP'),  # Top alignment for multi-line cells
            ]))
            
            # Alternate row colors
            for i in range(1, len(table_data)):
                bg_color = colors.HexColor('#E6E6E6') if i % 2 == 0 else colors.white
                table.setStyle(TableStyle(
                    [('BACKGROUND', (0, i), (-1, i), bg_color)]
                ))
            
            elements.append(table)
            
            # Build document with page number
            def add_page_num(canvas, doc):
                canvas.saveState()
                canvas.setFont('Helvetica', 8)
                page_num = f"Page {doc.page}"
                canvas.drawRightString(
                    CUSTOM_WIDTH - 20,
                    20,
                    page_num
                )
                canvas.restoreState()
            
            doc.build(elements, onFirstPage=add_page_num, onLaterPages=add_page_num)
            logger.info(f"Wide-format PDF generated: {filepath}")
            
        except Exception as e:
            logger.error(f"PDF generation error: {str(e)}")
            raise
if __name__ == "__main__":
    # Usage example
    service = PrixNcExctractService(bash_size=1000)
    
    if (raw_products := service.load_data()):
        exporter = JSONExporter(service.cleaning_data(raw_products))
        exporter.to_pdf("produits_nc.pdf", "Here is the given title")
    else:
        logger.error("Data loading failed")