"""================================================================================
SEC EDGAR DATA CLIENT - Insider Trading & Institutional Holdings
================================================================================
Author: Tom Hogan | Alpha Loop Capital, LLC

SEC EDGAR provides FREE access to critical regulatory filings that reveal
what insiders and institutions are ACTUALLY doing with their money.

KEY FILINGS WE TRACK:
- Form 4: Insider transactions (buy/sell within 2 business days)
- Form 13F: Institutional holdings (quarterly, 45 day lag)
- Form 13D: Activist positions (>5% ownership)
- Form 13G: Passive positions (>5% ownership)
- Form 8-K: Material events
- DEF 14A: Proxy statements (executive compensation)

PHILOSOPHY:
"Watch what they do, not what they say."
- Insiders buy for ONE reason: they think the stock is going up
- Insiders sell for MANY reasons: diversification, taxes, divorce, etc.
- Cluster buying is the strongest signal

WHY THIS MATTERS:
- Insiders have information we don't
- 13F shows what the smart money actually owns
- Form 4 is the closest thing to legal insider info

API: https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany
Rate Limit: 10 requests/second (be a good citizen)
================================================================================
"""

import hashlib
import logging
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional

try:
    import requests
except ImportError:
    requests = None

logger = logging.getLogger(__name__)


class InsiderTransactionType(Enum):
    """Types of insider transactions."""

    PURCHASE = "P"  # Open market purchase
    SALE = "S"  # Open market sale
    AWARD = "A"  # Award (grant)
    EXERCISE = "M"  # Exercise of derivative
    GIFT = "G"  # Gift
    CONVERSION = "C"  # Conversion


class InsiderRole(Enum):
    """Insider roles."""

    CEO = "CEO"
    CFO = "CFO"
    COO = "COO"
    DIRECTOR = "Director"
    OFFICER = "Officer"
    TEN_PERCENT_OWNER = "10% Owner"
    OTHER = "Other"


@dataclass
class InsiderTransaction:
    """A single insider transaction from Form 4."""

    transaction_id: str
    filing_date: datetime
    transaction_date: datetime

    # Company info
    ticker: str
    company_name: str
    cik: str  # SEC Central Index Key

    # Insider info
    insider_name: str
    insider_cik: str
    insider_role: InsiderRole
    is_officer: bool
    is_director: bool
    is_ten_percent_owner: bool

    # Transaction details
    transaction_type: InsiderTransactionType
    shares: int
    price_per_share: float
    total_value: float

    # Holdings after
    shares_owned_after: int
    ownership_nature: str  # "direct" or "indirect"

    # Our analysis
    signal_strength: str  # "strong", "moderate", "weak"
    cluster_member: bool  # Part of cluster buying/selling?

    def to_dict(self) -> Dict:
        return {
            "transaction_id": self.transaction_id,
            "filing_date": self.filing_date.isoformat(),
            "transaction_date": self.transaction_date.isoformat(),
            "ticker": self.ticker,
            "company_name": self.company_name,
            "insider_name": self.insider_name,
            "insider_role": self.insider_role.value,
            "transaction_type": self.transaction_type.value,
            "shares": self.shares,
            "price": self.price_per_share,
            "total_value": self.total_value,
            "shares_owned_after": self.shares_owned_after,
            "signal_strength": self.signal_strength,
            "is_cluster": self.cluster_member,
        }


@dataclass
class InstitutionalHolding:
    """An institutional holding from Form 13F."""

    holding_id: str
    filing_date: datetime
    report_date: datetime  # Quarter end date

    # Institution info
    institution_name: str
    institution_cik: str
    institution_type: str  # "hedge_fund", "mutual_fund", "pension", etc.

    # Position info
    ticker: str
    company_name: str
    cusip: str

    shares: int
    value: float  # In thousands, as reported

    # Changes
    shares_change: int
    shares_change_pct: float
    is_new_position: bool
    is_exit: bool

    # Our analysis
    conviction_level: str  # Based on position size vs AUM
    smart_money_score: float  # How "smart" is this institution?

    def to_dict(self) -> Dict:
        return {
            "holding_id": self.holding_id,
            "filing_date": self.filing_date.isoformat(),
            "report_date": self.report_date.isoformat(),
            "institution": self.institution_name,
            "ticker": self.ticker,
            "shares": self.shares,
            "value_thousands": self.value,
            "shares_change": self.shares_change,
            "shares_change_pct": self.shares_change_pct,
            "is_new": self.is_new_position,
            "is_exit": self.is_exit,
            "conviction": self.conviction_level,
            "smart_money_score": self.smart_money_score,
        }


@dataclass
class ActivistPosition:
    """An activist position from Form 13D."""

    position_id: str
    filing_date: datetime

    # Activist info
    activist_name: str
    activist_cik: str
    is_known_activist: bool  # Icahn, Ackman, etc.

    # Position info
    ticker: str
    company_name: str

    ownership_pct: float
    shares: int
    average_cost: float

    # Intent
    stated_intent: str  # Parsed from filing
    is_hostile: bool
    seeking_board_seats: bool
    seeking_sale: bool

    def to_dict(self) -> Dict:
        return {
            "position_id": self.position_id,
            "filing_date": self.filing_date.isoformat(),
            "activist": self.activist_name,
            "ticker": self.ticker,
            "ownership_pct": self.ownership_pct,
            "shares": self.shares,
            "is_known_activist": self.is_known_activist,
            "is_hostile": self.is_hostile,
            "seeking_board_seats": self.seeking_board_seats,
        }


class SECEdgarClient:
    """SEC EDGAR Data Client

    Fetches and parses SEC filings for insider and institutional intelligence.

    RATE LIMITS:
    - 10 requests/second maximum
    - Use a reasonable User-Agent header
    - Don't scrape excessively

    API ENDPOINTS:
    - Company search: https://www.sec.gov/cgi-bin/browse-edgar
    - Full-text search: https://efts.sec.gov/LATEST/search-index
    - Filings: https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany

    USER SETUP:
    1. Get your User-Agent: "YourName yourname@email.com"
    2. Set ALPHA_LOOP_SEC_EMAIL environment variable
    """

    BASE_URL = "https://www.sec.gov"
    EDGAR_URL = "https://www.sec.gov/cgi-bin/browse-edgar"
    DATA_URL = "https://data.sec.gov"

    # Known activist investors to flag
    KNOWN_ACTIVISTS = [
        "icahn", "ackman", "loeb", "peltz", "singer", "ubben",
        "starboard", "elliott", "valueact", "jana", "trian",
        "third point", "pershing square", "greenlight",
    ]

    # Smart money institutions (historically good returns)
    SMART_MONEY = [
        "berkshire hathaway",
        "renaissance technologies",
        "two sigma",
        "bridgewater",
        "citadel",
        "millennium",
        "point72",
        "baupost",
        "appaloosa",
        "lone pine",
        "tiger global",
        "coatue",
        "viking",
        "d1 capital",
    ]

    def __init__(self, user_email: str = "alphaloop@example.com"):
        """Initialize SEC EDGAR client.

        Args:
        ----
            user_email: Your email for User-Agent header (SEC requirement)
        """
        self.user_email = user_email
        self.headers = {
            "User-Agent": f"Alpha Loop Capital {user_email}",
            "Accept-Encoding": "gzip, deflate",
        }

        # Rate limiting
        self.last_request_time = 0
        self.min_request_interval = 0.1  # 10 requests/second

        # Caches
        self.cik_cache: Dict[str, str] = {}  # ticker -> CIK
        self.company_cache: Dict[str, Dict] = {}

        # Transaction tracking
        self.insider_transactions: List[InsiderTransaction] = []
        self.institutional_holdings: List[InstitutionalHolding] = []
        self.activist_positions: List[ActivistPosition] = []

        self.logger = logging.getLogger(__name__)
        self.logger.info(f"SEC EDGAR client initialized with email: {user_email}")

    def _rate_limit(self):
        """Enforce rate limiting."""
        elapsed = time.time() - self.last_request_time
        if elapsed < self.min_request_interval:
            time.sleep(self.min_request_interval - elapsed)
        self.last_request_time = time.time()

    def _make_request(self, url: str, params: Dict = None) -> Optional[Any]:
        """Make rate-limited request to SEC."""
        if requests is None:
            self.logger.error("requests library not installed")
            return None

        self._rate_limit()

        try:
            response = requests.get(url, params=params, headers=self.headers, timeout=30)
            response.raise_for_status()
            return response
        except Exception as e:
            self.logger.error(f"SEC request failed: {e}")
            return None

    # =========================================================================
    # CIK LOOKUP - Get SEC Central Index Key for a ticker
    # =========================================================================

    def get_cik(self, ticker: str) -> Optional[str]:
        """Get CIK (Central Index Key) for a ticker.

        CIK is the unique identifier SEC uses for all filers.

        Args:
        ----
            ticker: Stock ticker symbol

        Returns:
        -------
            10-digit CIK string or None
        """
        ticker = ticker.upper()

        # Check cache
        if ticker in self.cik_cache:
            return self.cik_cache[ticker]

        # Use SEC's company tickers JSON
        url = f"{self.DATA_URL}/submissions/CIK{ticker}.json"
        response = self._make_request(url)

        if response is None:
            # Try company search
            return self._search_cik(ticker)

        try:
            data = response.json()
            cik = str(data.get("cik", "")).zfill(10)
            self.cik_cache[ticker] = cik
            return cik
        except Exception as e:
            self.logger.error(f"Error parsing CIK response: {e}")
            return self._search_cik(ticker)

    def _search_cik(self, ticker: str) -> Optional[str]:
        """Search for CIK using company search."""
        # Use ticker mapping JSON from SEC
        url = f"{self.DATA_URL}/company_tickers.json"
        response = self._make_request(url)

        if response is None:
            return None

        try:
            data = response.json()
            for entry in data.values():
                if entry.get("ticker", "").upper() == ticker.upper():
                    cik = str(entry.get("cik_str", "")).zfill(10)
                    self.cik_cache[ticker] = cik
                    return cik
        except Exception as e:
            self.logger.error(f"Error searching CIK: {e}")

        return None

    # =========================================================================
    # FORM 4 - INSIDER TRANSACTIONS
    # =========================================================================

    def get_insider_transactions(
        self,
        ticker: str,
        days_back: int = 90,
    ) -> List[InsiderTransaction]:
        """Get recent insider transactions for a company.

        FORM 4 must be filed within 2 business days of transaction.

        WHAT TO LOOK FOR:
        - Cluster buying (multiple insiders buying)
        - Large purchases relative to salary
        - CEO/CFO purchases (highest conviction)
        - Purchases after stock decline (contrarian insiders)

        Args:
        ----
            ticker: Stock ticker
            days_back: How many days of history

        Returns:
        -------
            List of insider transactions
        """
        cik = self.get_cik(ticker)
        if not cik:
            self.logger.warning(f"Could not find CIK for {ticker}")
            return []

        # Fetch recent filings
        url = f"{self.DATA_URL}/submissions/CIK{cik}.json"
        response = self._make_request(url)

        if response is None:
            return []

        try:
            data = response.json()
            company_name = data.get("name", ticker)
            recent_filings = data.get("filings", {}).get("recent", {})

            transactions = []
            cutoff_date = datetime.now() - timedelta(days=days_back)

            form_types = recent_filings.get("form", [])
            filing_dates = recent_filings.get("filingDate", [])
            accession_numbers = recent_filings.get("accessionNumber", [])

            for i, form_type in enumerate(form_types):
                if form_type not in ["4", "4/A"]:
                    continue

                filing_date = datetime.strptime(filing_dates[i], "%Y-%m-%d")
                if filing_date < cutoff_date:
                    continue

                # Parse Form 4 details
                accession = accession_numbers[i].replace("-", "")
                form4_url = f"{self.DATA_URL}/Archives/edgar/data/{cik}/{accession}/form4.xml"

                form4_transactions = self._parse_form4(
                    form4_url, ticker, company_name, cik, filing_date,
                )
                transactions.extend(form4_transactions)

            # Store and return
            self.insider_transactions.extend(transactions)

            # Detect clusters
            self._detect_cluster_buying(transactions, ticker)

            return transactions

        except Exception as e:
            self.logger.error(f"Error fetching insider transactions: {e}")
            return []

    def _parse_form4(
        self,
        url: str,
        ticker: str,
        company_name: str,
        cik: str,
        filing_date: datetime,
    ) -> List[InsiderTransaction]:
        """Parse Form 4 XML."""
        response = self._make_request(url)
        if response is None:
            return []

        transactions = []

        try:
            # Parse XML
            root = ET.fromstring(response.content)

            # Get insider info
            owner = root.find(".//reportingOwner")
            if owner is None:
                return []

            insider_name = ""
            owner_name = owner.find(".//rptOwnerName")
            if owner_name is not None:
                insider_name = owner_name.text or ""

            insider_cik = ""
            owner_cik = owner.find(".//rptOwnerCik")
            if owner_cik is not None:
                insider_cik = owner_cik.text or ""

            # Get relationship
            relationship = owner.find(".//reportingOwnerRelationship")
            is_officer = False
            is_director = False
            is_ten_pct = False
            officer_title = ""

            if relationship is not None:
                is_officer = relationship.find("isOfficer") is not None and relationship.find("isOfficer").text == "1"
                is_director = relationship.find("isDirector") is not None and relationship.find("isDirector").text == "1"
                is_ten_pct = relationship.find("isTenPercentOwner") is not None and relationship.find("isTenPercentOwner").text == "1"

                title = relationship.find("officerTitle")
                if title is not None:
                    officer_title = title.text or ""

            # Determine role
            if "ceo" in officer_title.lower() or "chief executive" in officer_title.lower():
                role = InsiderRole.CEO
            elif "cfo" in officer_title.lower() or "chief financial" in officer_title.lower():
                role = InsiderRole.CFO
            elif "coo" in officer_title.lower() or "chief operating" in officer_title.lower():
                role = InsiderRole.COO
            elif is_director:
                role = InsiderRole.DIRECTOR
            elif is_officer:
                role = InsiderRole.OFFICER
            elif is_ten_pct:
                role = InsiderRole.TEN_PERCENT_OWNER
            else:
                role = InsiderRole.OTHER

            # Parse transactions
            for txn in root.findall(".//nonDerivativeTransaction"):
                tx_date_elem = txn.find(".//transactionDate/value")
                if tx_date_elem is None:
                    continue

                tx_date = datetime.strptime(tx_date_elem.text, "%Y-%m-%d")

                # Transaction type
                tx_code_elem = txn.find(".//transactionCoding/transactionCode")
                tx_code = tx_code_elem.text if tx_code_elem is not None else "P"

                try:
                    tx_type = InsiderTransactionType(tx_code)
                except ValueError:
                    tx_type = InsiderTransactionType.PURCHASE

                # Shares
                shares_elem = txn.find(".//transactionAmounts/transactionShares/value")
                shares = int(float(shares_elem.text)) if shares_elem is not None else 0

                # Price
                price_elem = txn.find(".//transactionAmounts/transactionPricePerShare/value")
                price = float(price_elem.text) if price_elem is not None else 0.0

                # Shares owned after
                after_elem = txn.find(".//postTransactionAmounts/sharesOwnedFollowingTransaction/value")
                shares_after = int(float(after_elem.text)) if after_elem is not None else 0

                # Ownership nature
                ownership_elem = txn.find(".//ownershipNature/directOrIndirectOwnership/value")
                ownership = ownership_elem.text if ownership_elem is not None else "D"
                ownership_nature = "direct" if ownership == "D" else "indirect"

                # Calculate signal strength
                total_value = shares * price
                if tx_type == InsiderTransactionType.PURCHASE:
                    if total_value > 1000000:
                        signal_strength = "strong"
                    elif total_value > 100000:
                        signal_strength = "moderate"
                    else:
                        signal_strength = "weak"
                elif tx_type == InsiderTransactionType.SALE:
                    signal_strength = "weak"  # Sales are noisy
                else:
                    signal_strength = "neutral"

                transaction = InsiderTransaction(
                    transaction_id=f"txn_{hashlib.sha256(f'{ticker}{insider_name}{tx_date}'.encode()).hexdigest()[:8]}",
                    filing_date=filing_date,
                    transaction_date=tx_date,
                    ticker=ticker,
                    company_name=company_name,
                    cik=cik,
                    insider_name=insider_name,
                    insider_cik=insider_cik,
                    insider_role=role,
                    is_officer=is_officer,
                    is_director=is_director,
                    is_ten_percent_owner=is_ten_pct,
                    transaction_type=tx_type,
                    shares=shares,
                    price_per_share=price,
                    total_value=total_value,
                    shares_owned_after=shares_after,
                    ownership_nature=ownership_nature,
                    signal_strength=signal_strength,
                    cluster_member=False,
                )

                transactions.append(transaction)

        except Exception as e:
            self.logger.error(f"Error parsing Form 4: {e}")

        return transactions

    def _detect_cluster_buying(self, transactions: List[InsiderTransaction], ticker: str):
        """Detect cluster buying - multiple insiders buying within a short period.

        CLUSTER BUYING is the strongest insider signal because it indicates
        multiple people with inside information independently deciding to buy.
        """
        # Get purchases only
        purchases = [t for t in transactions if t.transaction_type == InsiderTransactionType.PURCHASE]

        if len(purchases) < 2:
            return

        # Sort by date
        purchases.sort(key=lambda x: x.transaction_date)

        # Look for clusters (3+ purchases within 30 days)
        for i, txn in enumerate(purchases):
            cluster_count = 0
            cluster_window = timedelta(days=30)

            for other in purchases:
                if other.transaction_id == txn.transaction_id:
                    continue

                if abs((txn.transaction_date - other.transaction_date).days) <= 30:
                    cluster_count += 1

            if cluster_count >= 2:  # 3+ total purchases
                txn.cluster_member = True
                txn.signal_strength = "strong"
                self.logger.info(f"CLUSTER BUYING detected for {ticker}: {cluster_count + 1} insiders buying")

    # =========================================================================
    # FORM 13F - INSTITUTIONAL HOLDINGS
    # =========================================================================

    def get_institutional_holdings(
        self,
        ticker: str,
        quarters_back: int = 4,
    ) -> List[InstitutionalHolding]:
        """Get institutional holdings for a stock.

        Form 13F is filed quarterly by institutions with >$100M AUM.
        45 day lag after quarter end.

        WHAT TO LOOK FOR:
        - Smart money (Renaissance, Bridgewater) increasing positions
        - Hedge funds building new positions
        - Mutual fund concentration
        - Position changes vs previous quarter

        Args:
        ----
            ticker: Stock ticker
            quarters_back: How many quarters of history

        Returns:
        -------
            List of institutional holdings
        """
        # Note: Full 13F parsing is complex and requires additional infrastructure
        # This is a simplified version that shows the structure

        cik = self.get_cik(ticker)
        if not cik:
            return []

        # In production, would fetch from SEC EDGAR 13F filings
        # For now, return empty list (would need more complex parsing)

        self.logger.info(f"13F holdings for {ticker} would be fetched here")
        return []

    def get_smart_money_positions(self, ticker: str) -> List[Dict]:
        """Get positions from known "smart money" institutions.

        These are funds with historically good returns.
        """
        smart_money_positions = []

        for holding in self.institutional_holdings:
            if holding.ticker != ticker:
                continue

            # Check if smart money
            inst_lower = holding.institution_name.lower()
            is_smart = any(sm in inst_lower for sm in self.SMART_MONEY)

            if is_smart:
                smart_money_positions.append(holding.to_dict())

        return smart_money_positions

    # =========================================================================
    # FORM 13D - ACTIVIST POSITIONS
    # =========================================================================

    def get_activist_positions(self, ticker: str) -> List[ActivistPosition]:
        """Get activist positions for a stock (Form 13D).

        Form 13D is filed when someone acquires >5% with intent
        to influence management. This is a MAJOR signal.

        WHAT TO LOOK FOR:
        - Known activists (Icahn, Ackman, Loeb) taking positions
        - 13D vs 13G (13D = activist intent, 13G = passive)
        - Board seat demands
        - Sale process demands

        Args:
        ----
            ticker: Stock ticker

        Returns:
        -------
            List of activist positions
        """
        cik = self.get_cik(ticker)
        if not cik:
            return []

        # Fetch 13D filings
        url = f"{self.DATA_URL}/submissions/CIK{cik}.json"
        response = self._make_request(url)

        if response is None:
            return []

        positions = []

        try:
            data = response.json()
            recent_filings = data.get("filings", {}).get("recent", {})

            form_types = recent_filings.get("form", [])
            filing_dates = recent_filings.get("filingDate", [])

            for i, form_type in enumerate(form_types):
                if form_type not in ["SC 13D", "SC 13D/A"]:
                    continue

                filing_date = datetime.strptime(filing_dates[i], "%Y-%m-%d")

                # Note: Would need to parse the actual 13D filing for details
                # This shows the structure

                position = ActivistPosition(
                    position_id=f"13d_{hashlib.sha256(f'{ticker}{filing_date}'.encode()).hexdigest()[:8]}",
                    filing_date=filing_date,
                    activist_name="Unknown",  # Would parse from filing
                    activist_cik="",
                    is_known_activist=False,  # Would check against KNOWN_ACTIVISTS
                    ticker=ticker,
                    company_name=data.get("name", ticker),
                    ownership_pct=0.0,  # Would parse from filing
                    shares=0,
                    average_cost=0.0,
                    stated_intent="",
                    is_hostile=False,
                    seeking_board_seats=False,
                    seeking_sale=False,
                )

                positions.append(position)

        except Exception as e:
            self.logger.error(f"Error fetching activist positions: {e}")

        return positions

    # =========================================================================
    # SYNTHESIS - COMBINING SIGNALS
    # =========================================================================

    def get_insider_score(self, ticker: str) -> Dict[str, Any]:
        """Calculate composite insider score for a stock.

        SCORING:
        - Recent purchases: +points
        - Recent sales: -points (small)
        - Cluster buying: BIG +points
        - CEO/CFO purchases: +points
        - Large purchases: +points

        Returns score from -100 to +100
        """
        transactions = self.get_insider_transactions(ticker, days_back=90)

        if not transactions:
            return {
                "ticker": ticker,
                "score": 0,
                "signal": "neutral",
                "transactions_count": 0,
                "net_shares": 0,
                "cluster_buying": False,
            }

        score = 0
        net_shares = 0
        cluster_buying = False

        for txn in transactions:
            if txn.transaction_type == InsiderTransactionType.PURCHASE:
                # Base points for purchase
                score += 10
                net_shares += txn.shares

                # Bonus for C-suite
                if txn.insider_role in [InsiderRole.CEO, InsiderRole.CFO, InsiderRole.COO]:
                    score += 15

                # Bonus for large purchases
                if txn.total_value > 1000000:
                    score += 20
                elif txn.total_value > 500000:
                    score += 10
                elif txn.total_value > 100000:
                    score += 5

                # Big bonus for cluster
                if txn.cluster_member:
                    score += 25
                    cluster_buying = True

            elif txn.transaction_type == InsiderTransactionType.SALE:
                # Small penalty for sales (they're noisy)
                score -= 3
                net_shares -= txn.shares

        # Normalize to -100 to +100
        score = max(-100, min(100, score))

        # Determine signal
        if score > 50:
            signal = "strong_buy"
        elif score > 20:
            signal = "buy"
        elif score > -20:
            signal = "neutral"
        elif score > -50:
            signal = "sell"
        else:
            signal = "strong_sell"

        return {
            "ticker": ticker,
            "score": score,
            "signal": signal,
            "transactions_count": len(transactions),
            "net_shares": net_shares,
            "cluster_buying": cluster_buying,
            "purchases": len([t for t in transactions if t.transaction_type == InsiderTransactionType.PURCHASE]),
            "sales": len([t for t in transactions if t.transaction_type == InsiderTransactionType.SALE]),
        }

    def get_ownership_summary(self, ticker: str) -> Dict[str, Any]:
        """Get complete ownership summary for a stock.

        Combines insider transactions, institutional holdings, and activist positions.
        """
        insider_score = self.get_insider_score(ticker)
        activist_positions = self.get_activist_positions(ticker)

        return {
            "ticker": ticker,
            "insider": insider_score,
            "activists": [p.to_dict() for p in activist_positions],
            "has_activist": len(activist_positions) > 0,
            "smart_money_positions": self.get_smart_money_positions(ticker),
        }


# =============================================================================
# SINGLETON
# =============================================================================

_sec_client: Optional[SECEdgarClient] = None


def get_sec_edgar_client(user_email: str = "alphaloop@example.com") -> SECEdgarClient:
    """Get SEC EDGAR client singleton."""
    global _sec_client
    if _sec_client is None:
        _sec_client = SECEdgarClient(user_email)
    return _sec_client
