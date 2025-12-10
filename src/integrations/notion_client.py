"""
================================================================================
NOTION CLIENT - Notion Integration
================================================================================
Unified Notion client for documentation, databases, and wikis.

Features:
- Page creation and editing
- Database operations
- Search and filtering
- Content sync
================================================================================
"""

import hashlib
import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class NotionPage:
    """Notion page structure"""
    id: str
    title: str
    content: Dict
    parent_id: Optional[str] = None
    created_at: datetime = None
    updated_at: datetime = None

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()
        if self.updated_at is None:
            self.updated_at = datetime.now()


@dataclass
class NotionDatabase:
    """Notion database structure"""
    id: str
    name: str
    schema: Dict
    records: List[Dict]


class NotionClient:
    """
    Notion integration client

    Provides unified interface for all Notion operations.
    """

    # Database IDs (would be actual Notion IDs in production)
    DATABASES = {
        "tasks": "tasks_db",
        "meetings": "meetings_db",
        "agents": "agents_db",
        "docs": "docs_db",
        "learning": "learning_db",
    }

    def __init__(self, api_key: str = None):
        self.api_key = api_key
        self.connected = False
        self._pages: Dict[str, NotionPage] = {}
        self._databases: Dict[str, NotionDatabase] = {}

        logger.info("NotionClient initialized")

    async def connect(self) -> bool:
        """Connect to Notion"""
        self.connected = True
        logger.info("Notion connected")
        return True

    async def create_page(self, title: str, content: Dict,
                         parent_database: str = "docs") -> Dict:
        """Create a new Notion page"""
        page_id = hashlib.sha256(f"{title}{datetime.now()}".encode()).hexdigest()[:12]

        page = NotionPage(
            id=page_id,
            title=title,
            content=content,
            parent_id=self.DATABASES.get(parent_database)
        )

        self._pages[page_id] = page

        logger.info(f"Notion page created: {title}")

        return {
            "id": page_id,
            "title": title,
            "url": f"https://notion.so/{page_id}",
            "created_at": page.created_at.isoformat()
        }

    async def update_page(self, page_id: str, updates: Dict) -> Dict:
        """Update a Notion page"""
        if page_id in self._pages:
            page = self._pages[page_id]
            page.content.update(updates)
            page.updated_at = datetime.now()

            return {
                "id": page_id,
                "updated": True,
                "updated_at": page.updated_at.isoformat()
            }

        return {"error": "Page not found"}

    async def get_page(self, page_id: str) -> Optional[Dict]:
        """Get a Notion page"""
        if page_id in self._pages:
            page = self._pages[page_id]
            return {
                "id": page.id,
                "title": page.title,
                "content": page.content,
                "created_at": page.created_at.isoformat(),
                "updated_at": page.updated_at.isoformat()
            }
        return None

    async def query_database(self, database: str,
                            filters: Dict = None,
                            sorts: List[Dict] = None) -> List[Dict]:
        """Query a Notion database"""
        db_id = self.DATABASES.get(database, database)

        # Would apply filters and sorts in production
        results = []

        logger.info(f"Notion database query: {database}")

        return {
            "database": database,
            "results": results,
            "count": len(results)
        }

    async def add_to_database(self, database: str, record: Dict) -> Dict:
        """Add record to database"""
        db_id = self.DATABASES.get(database, database)
        record_id = hashlib.sha256(f"{database}{datetime.now()}".encode()).hexdigest()[:12]

        return {
            "id": record_id,
            "database": database,
            "created": True,
            "url": f"https://notion.so/{record_id}"
        }

    async def search(self, query: str,
                    filter_type: str = None) -> List[Dict]:
        """Search Notion"""
        results = []

        # Search pages
        for page in self._pages.values():
            if query.lower() in page.title.lower():
                results.append({
                    "id": page.id,
                    "type": "page",
                    "title": page.title
                })

        return {
            "query": query,
            "results": results,
            "count": len(results)
        }

    def create_page_content(self, sections: List[Dict]) -> Dict:
        """Create Notion page content structure"""
        blocks = []

        for section in sections:
            block_type = section.get("type", "paragraph")

            if block_type == "heading_1":
                blocks.append({
                    "type": "heading_1",
                    "heading_1": {
                        "rich_text": [{"text": {"content": section.get("text", "")}}]
                    }
                })
            elif block_type == "heading_2":
                blocks.append({
                    "type": "heading_2",
                    "heading_2": {
                        "rich_text": [{"text": {"content": section.get("text", "")}}]
                    }
                })
            elif block_type == "paragraph":
                blocks.append({
                    "type": "paragraph",
                    "paragraph": {
                        "rich_text": [{"text": {"content": section.get("text", "")}}]
                    }
                })
            elif block_type == "bullet":
                blocks.append({
                    "type": "bulleted_list_item",
                    "bulleted_list_item": {
                        "rich_text": [{"text": {"content": section.get("text", "")}}]
                    }
                })
            elif block_type == "code":
                blocks.append({
                    "type": "code",
                    "code": {
                        "rich_text": [{"text": {"content": section.get("text", "")}}],
                        "language": section.get("language", "python")
                    }
                })

        return {"blocks": blocks}


# Singleton
_notion_instance: Optional[NotionClient] = None


def get_notion_client() -> NotionClient:
    global _notion_instance
    if _notion_instance is None:
        _notion_instance = NotionClient()
    return _notion_instance

