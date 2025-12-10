"""
================================================================================
BASE EXECUTIVE ASSISTANT - Permission-Controlled Personal/Professional Assistant
================================================================================
Authors: Tom Hogan (Founder & CIO) & Chris Friedman (COO)
Developer: Alpha Loop Capital, LLC

This module defines the base class for Executive Assistants with strict
permission controls. All assistants have READ-ONLY access by default and
require WRITTEN PERMISSION for any actions.

SECURITY MODEL:
- READ-ONLY access to local files and software by default
- NO actions without explicit written permission
- Permission tokens with expiration
- Full audit trail
- Separate personal and professional scopes
================================================================================
"""

import logging
import hashlib
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Set
from pathlib import Path

logger = logging.getLogger(__name__)


class PermissionLevel(Enum):
    """Permission levels for executive assistant actions."""
    NONE = auto()           # No access
    READ_ONLY = auto()      # Read-only access (default)
    READ_WRITE = auto()     # Read and write access
    EXECUTE = auto()        # Can execute actions
    FULL = auto()           # Full access (rare)


class AccessScope(Enum):
    """Scope of access for the assistant."""
    PERSONAL = auto()       # Personal matters (calendar, email, etc.)
    PROFESSIONAL = auto()   # Professional/business matters
    FINANCIAL = auto()      # Financial data and transactions
    LEGAL = auto()          # Legal documents and matters
    HEALTH = auto()         # Health-related information
    TRAVEL = auto()         # Travel arrangements
    COMMUNICATIONS = auto() # Communications (email, messages)
    FILES = auto()          # Local file system access
    SOFTWARE = auto()       # Software and applications
    RESEARCH = auto()       # Research and information gathering
    SCHEDULING = auto()     # Calendar and scheduling
    ALL = auto()            # All scopes (requires explicit grant)


@dataclass
class PermissionToken:
    """A token granting specific permission for a limited time."""
    token_id: str
    owner: str                          # TOM_HOGAN or CHRIS_FRIEDMAN
    granted_by: str                     # Who granted the permission
    scope: AccessScope
    level: PermissionLevel
    action_description: str             # What action is permitted
    granted_at: datetime = field(default_factory=datetime.now)
    expires_at: Optional[datetime] = None
    one_time_use: bool = False
    used: bool = False
    written_permission: str = ""        # The actual written permission text

    def is_valid(self) -> bool:
        """Check if this permission token is still valid."""
        if self.used and self.one_time_use:
            return False
        if self.expires_at and datetime.now() > self.expires_at:
            return False
        return True

    def use(self) -> bool:
        """Mark this token as used. Returns False if already invalid."""
        if not self.is_valid():
            return False
        self.used = True
        return True


@dataclass
class PermissionRequest:
    """A request for permission to perform an action."""
    request_id: str
    owner: str
    scope: AccessScope
    level_required: PermissionLevel
    action_description: str
    reason: str
    requested_at: datetime = field(default_factory=datetime.now)
    status: str = "pending"  # pending, approved, denied
    response: str = ""
    token: Optional[PermissionToken] = None


@dataclass
class AuditEntry:
    """An entry in the audit log."""
    timestamp: datetime
    owner: str
    action: str
    scope: AccessScope
    level: PermissionLevel
    permission_token_id: Optional[str]
    success: bool
    details: str


class BaseExecutiveAssistant(ABC):
    """
    Base class for Executive Assistants with strict permission controls.

    All executive assistants have:
    - READ-ONLY access to files and software by default
    - NO action capability without written permission
    - Full audit trail of all activities
    - Separate personal and professional scopes
    """

    def __init__(
        self,
        name: str,
        owner: str,
        owner_email: str
    ):
        self.name = name
        self.owner = owner
        self.owner_email = owner_email

        # Permission management
        self.active_permissions: Dict[str, PermissionToken] = {}
        self.pending_requests: Dict[str, PermissionRequest] = {}
        self.permission_history: List[PermissionToken] = []

        # Audit trail
        self.audit_log: List[AuditEntry] = []

        # Default access levels (READ-ONLY for everything)
        self.default_access: Dict[AccessScope, PermissionLevel] = {
            scope: PermissionLevel.READ_ONLY for scope in AccessScope
        }
        self.default_access[AccessScope.ALL] = PermissionLevel.NONE

        # Blocked paths (never accessible)
        self.blocked_paths: Set[str] = set()

        # Allowed read paths (can be customized)
        self.allowed_read_paths: Set[str] = set()

        logger.info(f"Executive Assistant {name} initialized for {owner}")
        logger.info("DEFAULT ACCESS: READ-ONLY | NO ACTIONS WITHOUT WRITTEN PERMISSION")

    # =========================================================================
    # PERMISSION MANAGEMENT
    # =========================================================================

    def request_permission(
        self,
        scope: AccessScope,
        level: PermissionLevel,
        action_description: str,
        reason: str
    ) -> PermissionRequest:
        """
        Request permission to perform an action.
        This does NOT grant permission - owner must approve with written permission.
        """
        request_id = self._generate_id("REQ")

        request = PermissionRequest(
            request_id=request_id,
            owner=self.owner,
            scope=scope,
            level_required=level,
            action_description=action_description,
            reason=reason
        )

        self.pending_requests[request_id] = request

        self._audit(
            action=f"Permission requested: {action_description}",
            scope=scope,
            level=level,
            success=True,
            details=f"Request ID: {request_id}, Reason: {reason}"
        )

        logger.info(f"Permission request created: {request_id}")
        logger.warning(f"AWAITING WRITTEN PERMISSION from {self.owner}")

        return request

    def grant_permission(
        self,
        request_id: str,
        written_permission: str,
        duration_hours: Optional[int] = None,
        one_time: bool = False
    ) -> Optional[PermissionToken]:
        """
        Grant permission based on a pending request.
        REQUIRES written_permission text from the owner.
        """
        if request_id not in self.pending_requests:
            logger.error(f"Permission request {request_id} not found")
            return None

        request = self.pending_requests[request_id]

        if not written_permission.strip():
            logger.error("WRITTEN PERMISSION REQUIRED - Cannot grant without written approval")
            return None

        # Create permission token
        token_id = self._generate_id("PERM")
        expires_at = None
        if duration_hours:
            expires_at = datetime.now() + timedelta(hours=duration_hours)

        token = PermissionToken(
            token_id=token_id,
            owner=self.owner,
            granted_by=self.owner,  # Must be granted by owner
            scope=request.scope,
            level=request.level_required,
            action_description=request.action_description,
            expires_at=expires_at,
            one_time_use=one_time,
            written_permission=written_permission
        )

        self.active_permissions[token_id] = token
        self.permission_history.append(token)

        # Update request status
        request.status = "approved"
        request.response = written_permission
        request.token = token
        del self.pending_requests[request_id]

        self._audit(
            action=f"Permission granted: {request.action_description}",
            scope=request.scope,
            level=request.level_required,
            permission_token_id=token_id,
            success=True,
            details=f"Written permission: {written_permission[:100]}..."
        )

        logger.info(f"Permission granted: {token_id}")
        return token

    def deny_permission(self, request_id: str, reason: str) -> bool:
        """Deny a permission request."""
        if request_id not in self.pending_requests:
            return False

        request = self.pending_requests[request_id]
        request.status = "denied"
        request.response = reason
        del self.pending_requests[request_id]

        self._audit(
            action=f"Permission denied: {request.action_description}",
            scope=request.scope,
            level=request.level_required,
            success=False,
            details=f"Denial reason: {reason}"
        )

        return True

    def revoke_permission(self, token_id: str) -> bool:
        """Revoke an active permission."""
        if token_id not in self.active_permissions:
            return False

        token = self.active_permissions[token_id]
        del self.active_permissions[token_id]

        self._audit(
            action=f"Permission revoked: {token.action_description}",
            scope=token.scope,
            level=token.level,
            permission_token_id=token_id,
            success=True,
            details="Permission revoked by owner"
        )

        return True

    def has_permission(
        self,
        scope: AccessScope,
        level: PermissionLevel
    ) -> tuple[bool, Optional[PermissionToken]]:
        """
        Check if permission exists for a specific scope and level.
        Returns (has_permission, token).
        """
        # Check default access first
        default = self.default_access.get(scope, PermissionLevel.NONE)
        if level.value <= default.value:
            return True, None

        # Check active permissions
        for token in self.active_permissions.values():
            if token.is_valid() and token.scope == scope and token.level.value >= level.value:
                return True, token

        return False, None

    # =========================================================================
    # READ-ONLY FILE ACCESS
    # =========================================================================

    def read_file(self, file_path: str) -> Dict[str, Any]:
        """
        Read a file (READ-ONLY access).
        This is the DEFAULT capability - no permission needed for reading allowed paths.
        """
        path = Path(file_path)

        # Check blocked paths
        for blocked in self.blocked_paths:
            if blocked in str(path):
                self._audit(
                    action=f"File read blocked: {file_path}",
                    scope=AccessScope.FILES,
                    level=PermissionLevel.READ_ONLY,
                    success=False,
                    details="Path is in blocked list"
                )
                return {
                    "success": False,
                    "error": "Access to this path is blocked",
                    "path": file_path
                }

        # Check if path is allowed or if we have permission
        has_perm, token = self.has_permission(AccessScope.FILES, PermissionLevel.READ_ONLY)

        if not has_perm:
            self._audit(
                action=f"File read denied: {file_path}",
                scope=AccessScope.FILES,
                level=PermissionLevel.READ_ONLY,
                success=False,
                details="No read permission"
            )
            return {
                "success": False,
                "error": "No read permission for this file",
                "path": file_path
            }

        try:
            if not path.exists():
                return {
                    "success": False,
                    "error": "File not found",
                    "path": file_path
                }

            if path.is_dir():
                # List directory contents
                contents = [str(p.name) for p in path.iterdir()]
                self._audit(
                    action=f"Directory listed: {file_path}",
                    scope=AccessScope.FILES,
                    level=PermissionLevel.READ_ONLY,
                    permission_token_id=token.token_id if token else None,
                    success=True,
                    details=f"Found {len(contents)} items"
                )
                return {
                    "success": True,
                    "type": "directory",
                    "path": file_path,
                    "contents": contents
                }
            else:
                # Read file contents
                content = path.read_text(encoding='utf-8', errors='ignore')
                self._audit(
                    action=f"File read: {file_path}",
                    scope=AccessScope.FILES,
                    level=PermissionLevel.READ_ONLY,
                    permission_token_id=token.token_id if token else None,
                    success=True,
                    details=f"Read {len(content)} characters"
                )
                return {
                    "success": True,
                    "type": "file",
                    "path": file_path,
                    "content": content,
                    "size": len(content)
                }

        except Exception as e:
            self._audit(
                action=f"File read error: {file_path}",
                scope=AccessScope.FILES,
                level=PermissionLevel.READ_ONLY,
                success=False,
                details=str(e)
            )
            return {
                "success": False,
                "error": str(e),
                "path": file_path
            }

    # =========================================================================
    # ACTION EXECUTION (REQUIRES WRITTEN PERMISSION)
    # =========================================================================

    def execute_action(
        self,
        action: str,
        scope: AccessScope,
        params: Dict[str, Any],
        permission_token_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Execute an action. REQUIRES WRITTEN PERMISSION via permission token.
        """
        # Check for valid permission
        if permission_token_id:
            if permission_token_id not in self.active_permissions:
                self._audit(
                    action=f"Action blocked: {action}",
                    scope=scope,
                    level=PermissionLevel.EXECUTE,
                    success=False,
                    details="Invalid permission token"
                )
                return {
                    "success": False,
                    "error": "Invalid permission token",
                    "action": action
                }

            token = self.active_permissions[permission_token_id]
            if not token.is_valid():
                self._audit(
                    action=f"Action blocked: {action}",
                    scope=scope,
                    level=PermissionLevel.EXECUTE,
                    permission_token_id=permission_token_id,
                    success=False,
                    details="Permission token expired or already used"
                )
                return {
                    "success": False,
                    "error": "Permission token is no longer valid",
                    "action": action
                }

            if token.scope != scope and token.scope != AccessScope.ALL:
                self._audit(
                    action=f"Action blocked: {action}",
                    scope=scope,
                    level=PermissionLevel.EXECUTE,
                    permission_token_id=permission_token_id,
                    success=False,
                    details="Permission token scope mismatch"
                )
                return {
                    "success": False,
                    "error": "Permission token does not cover this scope",
                    "action": action
                }

            # Mark token as used if one-time
            token.use()

            # Execute the action
            result = self._execute_permitted_action(action, scope, params)

            self._audit(
                action=f"Action executed: {action}",
                scope=scope,
                level=PermissionLevel.EXECUTE,
                permission_token_id=permission_token_id,
                success=result.get("success", False),
                details=str(params)[:200]
            )

            return result

        else:
            # No permission token - action denied
            self._audit(
                action=f"Action blocked: {action}",
                scope=scope,
                level=PermissionLevel.EXECUTE,
                success=False,
                details="NO WRITTEN PERMISSION PROVIDED"
            )
            return {
                "success": False,
                "error": "WRITTEN PERMISSION REQUIRED. No action taken.",
                "action": action,
                "hint": "Use request_permission() to request permission, then owner must grant with written approval"
            }

    @abstractmethod
    def _execute_permitted_action(
        self,
        action: str,
        scope: AccessScope,
        params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute a permitted action. Override in subclasses."""
        pass

    # =========================================================================
    # AUDIT AND LOGGING
    # =========================================================================

    def _audit(
        self,
        action: str,
        scope: AccessScope,
        level: PermissionLevel,
        permission_token_id: Optional[str] = None,
        success: bool = True,
        details: str = ""
    ):
        """Add an entry to the audit log."""
        entry = AuditEntry(
            timestamp=datetime.now(),
            owner=self.owner,
            action=action,
            scope=scope,
            level=level,
            permission_token_id=permission_token_id,
            success=success,
            details=details
        )
        self.audit_log.append(entry)

        # Also log to standard logger
        level_str = "INFO" if success else "WARNING"
        logger.log(
            logging.INFO if success else logging.WARNING,
            f"[AUDIT] {self.name} | {action} | {scope.name} | {success} | {details[:100]}"
        )

    def get_audit_log(
        self,
        since: Optional[datetime] = None,
        scope: Optional[AccessScope] = None
    ) -> List[Dict[str, Any]]:
        """Get audit log entries, optionally filtered."""
        entries = self.audit_log

        if since:
            entries = [e for e in entries if e.timestamp >= since]

        if scope:
            entries = [e for e in entries if e.scope == scope]

        return [
            {
                "timestamp": e.timestamp.isoformat(),
                "owner": e.owner,
                "action": e.action,
                "scope": e.scope.name,
                "level": e.level.name,
                "permission_token_id": e.permission_token_id,
                "success": e.success,
                "details": e.details
            }
            for e in entries
        ]

    # =========================================================================
    # UTILITIES
    # =========================================================================

    def _generate_id(self, prefix: str) -> str:
        """Generate a unique ID."""
        timestamp = datetime.now().isoformat()
        data = f"{prefix}-{self.owner}-{timestamp}"
        hash_val = hashlib.sha256(data.encode()).hexdigest()[:12]
        return f"{prefix}-{hash_val}"

    def get_status(self) -> Dict[str, Any]:
        """Get current status of the executive assistant."""
        return {
            "name": self.name,
            "owner": self.owner,
            "active_permissions": len(self.active_permissions),
            "pending_requests": len(self.pending_requests),
            "audit_entries": len(self.audit_log),
            "default_access": {k.name: v.name for k, v in self.default_access.items()},
            "blocked_paths": len(self.blocked_paths)
        }

    def get_pending_requests(self) -> List[Dict[str, Any]]:
        """Get all pending permission requests."""
        return [
            {
                "request_id": r.request_id,
                "scope": r.scope.name,
                "level_required": r.level_required.name,
                "action_description": r.action_description,
                "reason": r.reason,
                "requested_at": r.requested_at.isoformat()
            }
            for r in self.pending_requests.values()
        ]

    def get_active_permissions(self) -> List[Dict[str, Any]]:
        """Get all active permissions."""
        return [
            {
                "token_id": t.token_id,
                "scope": t.scope.name,
                "level": t.level.name,
                "action_description": t.action_description,
                "expires_at": t.expires_at.isoformat() if t.expires_at else None,
                "one_time_use": t.one_time_use,
                "used": t.used,
                "is_valid": t.is_valid()
            }
            for t in self.active_permissions.values()
        ]

    @abstractmethod
    def get_capabilities(self) -> List[str]:
        """Get list of capabilities. Override in subclasses."""
        pass

    @abstractmethod
    def get_natural_language_explanation(self) -> str:
        """Get explanation of what this assistant does. Override in subclasses."""
        pass

