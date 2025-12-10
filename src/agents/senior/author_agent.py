"""
================================================================================
THE_AUTHOR AGENT - Natural Language Writer in Tom's Voice
================================================================================
Author: Tom Hogan
Developer: Alpha Loop Capital, LLC

THE_AUTHOR writes in natural language that mimics how Tom Hogan writes.
Reference style sources:
- tomhoganfinance.substack.com
- @hoags18 on X (Twitter)
- alcresearch.substack.com

Outputs:
- Notion docs
- Word docs  
- Training report summaries
- Substack drafts (personal and ALC)
- Twitter/X posts
- Professional market/stock analysis
- Agent update reports

All docs are logged and tracked.

Tier: SENIOR (2)
Reports To: HOAGS → Tom
Cluster: content

Core Philosophy:
"Write like Tom - direct, analytical, contrarian, with dry humor."
================================================================================
"""

import logging
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from enum import Enum

from src.core.agent_base import BaseAgent, AgentTier

logger = logging.getLogger(__name__)


class ContentType(Enum):
    """Types of content THE_AUTHOR produces"""
    TRAINING_SUMMARY = "training_summary"
    SUBSTACK_PERSONAL = "substack_personal"
    SUBSTACK_ALC = "substack_alc"
    TWITTER_POST = "twitter_post"
    TWITTER_THREAD = "twitter_thread"
    NOTION_DOC = "notion_doc"
    WORD_DOC = "word_doc"
    AGENT_UPDATE = "agent_update"
    MARKET_ANALYSIS = "market_analysis"
    STOCK_WRITEUP = "stock_writeup"
    INTERNAL_MEMO = "internal_memo"


class ToneStyle(Enum):
    """Writing tone variations"""
    ANALYTICAL = "analytical"       # Data-driven, precise
    CONTRARIAN = "contrarian"       # Challenging consensus
    HUMOROUS = "humorous"           # Dry wit, sarcastic
    URGENT = "urgent"               # Time-sensitive alerts
    EDUCATIONAL = "educational"     # Explaining concepts
    PROFESSIONAL = "professional"   # Formal, institutional


@dataclass
class WrittenDocument:
    """A document produced by THE_AUTHOR"""
    doc_id: str
    content_type: ContentType
    tone: ToneStyle
    title: str
    content: str
    word_count: int
    target_platform: str
    
    # Metadata
    created_at: datetime = field(default_factory=datetime.now)
    last_edited: datetime = field(default_factory=datetime.now)
    version: int = 1
    is_draft: bool = True
    
    # Distribution
    published: bool = False
    published_at: Optional[datetime] = None
    published_url: Optional[str] = None
    
    # Analytics
    engagement_metrics: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        return {
            "doc_id": self.doc_id,
            "type": self.content_type.value,
            "tone": self.tone.value,
            "title": self.title,
            "content": self.content[:500] + "..." if len(self.content) > 500 else self.content,
            "word_count": self.word_count,
            "platform": self.target_platform,
            "created_at": self.created_at.isoformat(),
            "version": self.version,
            "is_draft": self.is_draft,
            "published": self.published
        }


@dataclass
class AgentUpdateReport:
    """Report on agent improvements and changes"""
    report_id: str
    timestamp: datetime
    agents_updated: List[str]
    improvements: List[Dict[str, Any]]
    performance_changes: Dict[str, float]
    narrative_summary: str
    
    def to_dict(self) -> Dict:
        return {
            "report_id": self.report_id,
            "timestamp": self.timestamp.isoformat(),
            "agents_updated": self.agents_updated,
            "improvements": self.improvements,
            "performance_changes": self.performance_changes,
            "summary": self.narrative_summary
        }


class TheAuthorAgent(BaseAgent):
    """
    THE_AUTHOR Agent - Writing in Tom's Voice
    
    THE_AUTHOR mimics Tom Hogan's writing style across all content types.
    
    Tom's Writing Characteristics:
    - Direct and concise, no fluff
    - Data-driven with specific numbers
    - Contrarian takes on consensus views
    - Dry humor and occasional sarcasm
    - Uses questions to make points
    - References specific trades/positions
    - Honest about uncertainty
    - Explains the "why" behind ideas
    
    Key Methods:
    - write(): Main writing function
    - summarize_training(): Create training report summaries
    - draft_substack(): Write Substack articles
    - compose_tweet(): Write tweets/threads
    - document_agent_updates(): Track agent improvements
    """
    
    # Tom's signature phrases and patterns
    TOM_PATTERNS = {
        "transitions": [
            "Here's the thing:",
            "The question is:",
            "Look,",
            "Bottom line:",
            "What's interesting here:",
            "The key insight:",
        ],
        "contrarian_markers": [
            "Everyone thinks X, but",
            "The consensus is wrong because",
            "What the market is missing:",
            "Counterintuitively,",
            "The dog that didn't bark:",
        ],
        "data_references": [
            "The numbers tell a different story:",
            "Looking at the data:",
            "Here's what matters:",
            "The math is simple:",
        ],
        "humor_inserts": [
            "(yes, really)",
            "- shocking, I know",
            "spoiler alert:",
            "plot twist:",
            "*narrator voice* it was not, in fact,",
        ],
        "closings": [
            "We'll see how this plays out.",
            "Time will tell.",
            "More to come.",
            "Stay tuned.",
            "NFA, obviously.",
        ]
    }
    
    def __init__(self):
        super().__init__(
            name="THE_AUTHOR",
            tier=AgentTier.SENIOR,
            capabilities=[
                # Core writing
                "natural_language_generation",
                "style_mimicry",
                "tone_adaptation",
                
                # Content types
                "training_summaries",
                "substack_drafting",
                "twitter_composition",
                "market_analysis_writing",
                "stock_writeups",
                
                # Documentation
                "notion_integration",
                "word_doc_generation",
                "version_control",
                
                # Agent tracking
                "agent_update_documentation",
                "performance_narratives",
                
                # Quality
                "grammar_checking",
                "fact_verification",
                "tone_consistency"
            ],
            user_id="TJH"
        )
        
        # Document storage
        self.documents: List[WrittenDocument] = []
        self.agent_reports: List[AgentUpdateReport] = []
        
        # Statistics
        self.total_words_written = 0
        self.documents_published = 0
        
        # Style settings
        self.default_tone = ToneStyle.ANALYTICAL
        self.use_humor = True
        self.max_tweet_length = 280
        self.max_thread_tweets = 15
    
    def process(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Process a writing task"""
        action = task.get("action", task.get("type", ""))
        params = task.get("parameters", task)
        
        self.log_action(action, f"THE_AUTHOR processing: {action}")
        
        # Check for capability gaps (ACA)
        gap = self.detect_capability_gap(task)
        if gap:
            self.logger.warning(f"Capability gap: {gap.missing_capabilities}")
        
        handlers = {
            "write": self._handle_write,
            "summarize_training": self._handle_summarize_training,
            "draft_substack": self._handle_draft_substack,
            "compose_tweet": self._handle_compose_tweet,
            "compose_thread": self._handle_compose_thread,
            "document_agents": self._handle_document_agents,
            "create_notion": self._handle_create_notion,
            "get_documents": self._handle_get_documents,
            "publish": self._handle_publish,
        }
        
        handler = handlers.get(action, self._handle_unknown)
        return handler(params)
    
    def get_capabilities(self) -> List[str]:
        return self.capabilities
    
    # =========================================================================
    # CORE WRITING METHODS
    # =========================================================================
    
    def write(
        self,
        content_type: ContentType,
        topic: str,
        data: Dict[str, Any] = None,
        tone: ToneStyle = None,
        max_words: int = None
    ) -> WrittenDocument:
        """
        Main writing function - produces content in Tom's style.
        
        Args:
            content_type: Type of content to produce
            topic: Subject matter
            data: Supporting data to incorporate
            tone: Writing tone (defaults to analytical)
            max_words: Word limit
        
        Returns:
            WrittenDocument object
        """
        import hashlib
        import random
        
        tone = tone or self.default_tone
        
        # Generate content based on type
        if content_type == ContentType.TWITTER_POST:
            content = self._generate_tweet(topic, data, tone)
            max_words = 50
        elif content_type == ContentType.TWITTER_THREAD:
            content = self._generate_thread(topic, data, tone)
            max_words = 500
        elif content_type in [ContentType.SUBSTACK_PERSONAL, ContentType.SUBSTACK_ALC]:
            content = self._generate_substack(topic, data, tone, content_type)
            max_words = max_words or 1500
        elif content_type == ContentType.TRAINING_SUMMARY:
            content = self._generate_training_summary(topic, data)
            max_words = max_words or 500
        elif content_type == ContentType.AGENT_UPDATE:
            content = self._generate_agent_update(data)
            max_words = max_words or 300
        else:
            content = self._generate_general(topic, data, tone)
            max_words = max_words or 800
        
        # Create document
        doc = WrittenDocument(
            doc_id=f"doc_{hashlib.sha256(f'{topic}{datetime.now()}'.encode()).hexdigest()[:8]}",
            content_type=content_type,
            tone=tone,
            title=self._generate_title(topic, content_type),
            content=content,
            word_count=len(content.split()),
            target_platform=self._get_platform(content_type)
        )
        
        # Store and track
        self.documents.append(doc)
        self.total_words_written += doc.word_count
        
        self.logger.info(f"THE_AUTHOR: Created {content_type.value} - {doc.word_count} words")
        
        return doc
    
    def summarize_training(
        self,
        training_data: Dict[str, Any],
        include_metrics: bool = True
    ) -> WrittenDocument:
        """
        Create a summary of training results in Tom's style.
        """
        return self.write(
            content_type=ContentType.TRAINING_SUMMARY,
            topic="Training Results",
            data=training_data
        )
    
    def draft_substack(
        self,
        topic: str,
        thesis: str,
        data: Dict[str, Any] = None,
        for_alc: bool = False
    ) -> WrittenDocument:
        """
        Draft a Substack article.
        
        Args:
            topic: Article topic
            thesis: Main argument/thesis
            data: Supporting data
            for_alc: True for ALC Research, False for personal
        """
        content_type = ContentType.SUBSTACK_ALC if for_alc else ContentType.SUBSTACK_PERSONAL
        
        return self.write(
            content_type=content_type,
            topic=topic,
            data={"thesis": thesis, **(data or {})}
        )
    
    def compose_tweet(
        self,
        topic: str,
        key_point: str,
        include_data: bool = True
    ) -> WrittenDocument:
        """
        Compose a single tweet in Tom's style.
        """
        return self.write(
            content_type=ContentType.TWITTER_POST,
            topic=topic,
            data={"key_point": key_point, "include_data": include_data}
        )
    
    def document_agent_updates(
        self,
        agents_updated: List[str],
        improvements: List[Dict[str, Any]],
        performance_changes: Dict[str, float]
    ) -> AgentUpdateReport:
        """
        Document agent improvements and changes.
        """
        import hashlib
        
        # Generate narrative summary
        narrative = self._generate_agent_update({
            "agents": agents_updated,
            "improvements": improvements,
            "performance": performance_changes
        })
        
        report = AgentUpdateReport(
            report_id=f"rpt_{hashlib.sha256(str(datetime.now()).encode()).hexdigest()[:8]}",
            timestamp=datetime.now(),
            agents_updated=agents_updated,
            improvements=improvements,
            performance_changes=performance_changes,
            narrative_summary=narrative
        )
        
        self.agent_reports.append(report)
        
        # Also create as document
        self.write(
            content_type=ContentType.AGENT_UPDATE,
            topic="Agent Updates",
            data=report.to_dict()
        )
        
        return report
    
    # =========================================================================
    # CONTENT GENERATORS (Tom's Style)
    # =========================================================================
    
    def _generate_tweet(self, topic: str, data: Dict, tone: ToneStyle) -> str:
        """Generate tweet in Tom's style"""
        import random
        
        key_point = data.get("key_point", topic) if data else topic
        
        # Tom's tweet patterns
        patterns = [
            f"{key_point}. {random.choice(self.TOM_PATTERNS['closings'])}",
            f"{random.choice(self.TOM_PATTERNS['transitions'])} {key_point}",
            f"{key_point} {random.choice(self.TOM_PATTERNS['humor_inserts'])}",
        ]
        
        tweet = random.choice(patterns)
        
        # Ensure within limit
        if len(tweet) > self.max_tweet_length:
            tweet = tweet[:self.max_tweet_length - 3] + "..."
        
        return tweet
    
    def _generate_thread(self, topic: str, data: Dict, tone: ToneStyle) -> str:
        """Generate Twitter thread in Tom's style"""
        import random
        
        thread_parts = []
        
        # Hook tweet
        thread_parts.append(f"🧵 Thread on {topic}\n\nThis is important. Let me explain why:\n\n1/")
        
        # Main points (3-5 tweets)
        main_points = data.get("points", [topic]) if data else [topic]
        for i, point in enumerate(main_points[:5], 2):
            transition = random.choice(self.TOM_PATTERNS['transitions'])
            thread_parts.append(f"{transition} {point}\n\n{i}/")
        
        # Conclusion
        thread_parts.append(f"Bottom line: {topic} matters more than people think.\n\n{random.choice(self.TOM_PATTERNS['closings'])}")
        
        return "\n\n---\n\n".join(thread_parts)
    
    def _generate_substack(
        self,
        topic: str,
        data: Dict,
        tone: ToneStyle,
        content_type: ContentType
    ) -> str:
        """Generate Substack article in Tom's style"""
        import random
        
        thesis = data.get("thesis", f"Why {topic} matters") if data else f"Why {topic} matters"
        
        sections = []
        
        # Intro - hook the reader
        intro = f"""# {topic}

{random.choice(self.TOM_PATTERNS['contrarian_markers'])} let me explain.

{thesis}

"""
        sections.append(intro)
        
        # Body - the argument
        body = f"""{random.choice(self.TOM_PATTERNS['transitions'])}

{random.choice(self.TOM_PATTERNS['data_references'])}

The conventional wisdom says one thing. The data says another. I know which one I'm betting on.

**The Setup**

Most investors are looking at this wrong. They're focused on the headline numbers when the real story is in the details.

**What I'm Seeing**

[This is where the specific analysis goes - data points, charts, observations]

{random.choice(self.TOM_PATTERNS['humor_inserts'])}

**The Trade**

Based on this analysis, here's how I'm positioned...

"""
        sections.append(body)
        
        # Conclusion
        conclusion = f"""**Bottom Line**

{topic} is more important than consensus believes. The market is mispricing this.

{random.choice(self.TOM_PATTERNS['closings'])}

---

*This is not financial advice. Do your own research. NFA.*
"""
        sections.append(conclusion)
        
        return "".join(sections)
    
    def _generate_training_summary(self, topic: str, data: Dict) -> str:
        """Generate training summary"""
        import random
        
        metrics = data or {}
        
        summary = f"""## Training Report Summary

**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M')}

### Key Results

"""
        
        if metrics:
            for key, value in metrics.items():
                summary += f"- **{key}:** {value}\n"
        
        summary += f"""
### Analysis

{random.choice(self.TOM_PATTERNS['data_references'])}

The training run shows [improvement/regression] in key metrics. 

### Next Steps

1. [Action item 1]
2. [Action item 2]
3. [Action item 3]

{random.choice(self.TOM_PATTERNS['closings'])}
"""
        
        return summary
    
    def _generate_agent_update(self, data: Dict) -> str:
        """Generate agent update narrative"""
        agents = data.get("agents", [])
        improvements = data.get("improvements", [])
        performance = data.get("performance", {})
        
        narrative = f"""## Agent Update Report

**Timestamp:** {datetime.now().strftime('%Y-%m-%d %H:%M')}

### Agents Modified

"""
        
        for agent in agents:
            narrative += f"- {agent}\n"
        
        narrative += f"""
### Improvements Made

"""
        
        for imp in improvements:
            narrative += f"- {imp.get('description', str(imp))}\n"
        
        narrative += f"""
### Performance Impact

"""
        
        for metric, change in performance.items():
            direction = "↑" if change > 0 else "↓" if change < 0 else "→"
            narrative += f"- {metric}: {direction} {abs(change):.1%}\n"
        
        narrative += f"""
---

All changes logged and tracked. More updates to come.
"""
        
        return narrative
    
    def _generate_general(self, topic: str, data: Dict, tone: ToneStyle) -> str:
        """Generate general content"""
        import random
        
        return f"""{random.choice(self.TOM_PATTERNS['transitions'])}

{topic}

{random.choice(self.TOM_PATTERNS['data_references'])}

[Content here]

{random.choice(self.TOM_PATTERNS['closings'])}
"""
    
    def _generate_title(self, topic: str, content_type: ContentType) -> str:
        """Generate title for content"""
        if content_type == ContentType.TWITTER_POST:
            return f"Tweet: {topic[:30]}"
        elif content_type == ContentType.TWITTER_THREAD:
            return f"Thread: {topic[:30]}"
        elif content_type in [ContentType.SUBSTACK_PERSONAL, ContentType.SUBSTACK_ALC]:
            return topic
        else:
            return f"{content_type.value}: {topic[:40]}"
    
    def _get_platform(self, content_type: ContentType) -> str:
        """Get target platform for content type"""
        platforms = {
            ContentType.TWITTER_POST: "twitter",
            ContentType.TWITTER_THREAD: "twitter",
            ContentType.SUBSTACK_PERSONAL: "tomhoganfinance.substack.com",
            ContentType.SUBSTACK_ALC: "alcresearch.substack.com",
            ContentType.NOTION_DOC: "notion",
            ContentType.WORD_DOC: "word",
            ContentType.TRAINING_SUMMARY: "internal",
            ContentType.AGENT_UPDATE: "internal",
            ContentType.MARKET_ANALYSIS: "internal",
            ContentType.STOCK_WRITEUP: "internal",
        }
        return platforms.get(content_type, "internal")
    
    def log_action(self, action: str, description: str):
        """Log an action"""
        self.logger.info(f"[THE_AUTHOR] {action}: {description}")
    
    # =========================================================================
    # TASK HANDLERS
    # =========================================================================
    
    def _handle_write(self, params: Dict) -> Dict:
        content_type = ContentType(params.get("content_type", "notion_doc"))
        topic = params.get("topic", "")
        data = params.get("data")
        tone = ToneStyle(params.get("tone", "analytical")) if params.get("tone") else None
        
        doc = self.write(content_type, topic, data, tone)
        return {"status": "success", "document": doc.to_dict()}
    
    def _handle_summarize_training(self, params: Dict) -> Dict:
        doc = self.summarize_training(params.get("training_data", {}))
        return {"status": "success", "document": doc.to_dict()}
    
    def _handle_draft_substack(self, params: Dict) -> Dict:
        doc = self.draft_substack(
            topic=params.get("topic", ""),
            thesis=params.get("thesis", ""),
            data=params.get("data"),
            for_alc=params.get("for_alc", False)
        )
        return {"status": "success", "document": doc.to_dict()}
    
    def _handle_compose_tweet(self, params: Dict) -> Dict:
        doc = self.compose_tweet(
            topic=params.get("topic", ""),
            key_point=params.get("key_point", ""),
            include_data=params.get("include_data", True)
        )
        return {"status": "success", "document": doc.to_dict()}
    
    def _handle_compose_thread(self, params: Dict) -> Dict:
        doc = self.write(
            content_type=ContentType.TWITTER_THREAD,
            topic=params.get("topic", ""),
            data=params.get("data")
        )
        return {"status": "success", "document": doc.to_dict()}
    
    def _handle_document_agents(self, params: Dict) -> Dict:
        report = self.document_agent_updates(
            agents_updated=params.get("agents", []),
            improvements=params.get("improvements", []),
            performance_changes=params.get("performance", {})
        )
        return {"status": "success", "report": report.to_dict()}
    
    def _handle_create_notion(self, params: Dict) -> Dict:
        doc = self.write(
            content_type=ContentType.NOTION_DOC,
            topic=params.get("topic", ""),
            data=params.get("data")
        )
        return {"status": "success", "document": doc.to_dict()}
    
    def _handle_get_documents(self, params: Dict) -> Dict:
        content_type = params.get("content_type")
        docs = self.documents
        if content_type:
            docs = [d for d in docs if d.content_type.value == content_type]
        return {
            "status": "success",
            "documents": [d.to_dict() for d in docs[-20:]],
            "total": len(self.documents),
            "total_words": self.total_words_written
        }
    
    def _handle_publish(self, params: Dict) -> Dict:
        doc_id = params.get("doc_id")
        doc = next((d for d in self.documents if d.doc_id == doc_id), None)
        if doc:
            doc.published = True
            doc.published_at = datetime.now()
            doc.is_draft = False
            self.documents_published += 1
            return {"status": "success", "published": True, "doc_id": doc_id}
        return {"status": "error", "message": "Document not found"}
    
    def _handle_unknown(self, params: Dict) -> Dict:
        return {"status": "error", "message": "Unknown action"}


# =============================================================================
# SINGLETON
# =============================================================================

_author_instance: Optional[TheAuthorAgent] = None


def get_author() -> TheAuthorAgent:
    """Get THE_AUTHOR agent singleton"""
    global _author_instance
    if _author_instance is None:
        _author_instance = TheAuthorAgent()
    return _author_instance

