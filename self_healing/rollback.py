"""
Enhanced Rollback Manager with Model Versioning.

Provides comprehensive model version management with metadata tracking,
validation, audit trails, and integration with model registries.
"""

from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
from datetime import datetime
import joblib
import json
import shutil
import logging

from .config import RollbackConfig
from .exceptions import RollbackError, VersioningError


logger = logging.getLogger(__name__)


class RollbackManager:
    """
    Advanced rollback management with version control.
    
    Features:
    - Full model versioning with metadata
    - Tagged versions (stable, canary, latest)
    - Automatic cleanup of old versions
    - Validation after rollback
    - Audit logging for compliance
    - Model registry integration (optional)
    
    Args:
        config: RollbackConfig instance
        models_dir: Base directory for model storage
    """
    
    def __init__(
        self,
        config: Optional[RollbackConfig] = None,
        models_dir: str = "models/baseline"
    ):
        if config is None:
            config = RollbackConfig()
        
        self.config = config
        self.models_dir = Path(models_dir)
        self.versions_dir = self.models_dir / "versions"
        self.versions_dir.mkdir(parents=True, exist_ok=True)
        
        # Load version registry
        self.registry_path = self.models_dir / "version_registry.json"
        self.registry = self._load_registry()
        
        # Audit log
        self.audit_log_path = self.models_dir / "rollback_audit.json"
        self.audit_log = self._load_audit_log()
        
        logger.info(
            "RollbackManager initialized: %d versions tracked, max_versions=%d",
            len(self.registry), config.max_versions_to_keep
        )
    
    def save_version(
        self,
        model: Any,
        version_tag: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        set_as_latest: bool = True
    ) -> str:
        """
        Save a new model version.
        
        Args:
            model: Model to save
            version_tag: Optional tag (e.g., "stable", "v1.2.3")
            metadata: Optional metadata dictionary
            set_as_latest: Whether to set this as the latest version
        
        Returns:
            Version identifier
        
        Raises:
            VersioningError: If versioning fails
        """
        try:
            # Generate version ID
            version_id = f"v_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            version_path = self.versions_dir / f"{version_id}.joblib"
            
            # Save model
            joblib.dump(model, version_path)
            
            # Create metadata
            version_metadata = {
                'version_id': version_id,
                'timestamp': datetime.now().isoformat(),
                'model_type': type(model).__name__,
                'path': str(version_path),
                'tag': version_tag,
                'is_latest': set_as_latest,
                'custom_metadata': metadata or {}
            }
            
            # Save metadata
            metadata_path = self.versions_dir / f"{version_id}_metadata.json"
            with open(metadata_path, 'w') as f:
                json.dump(version_metadata, f, indent=2)
            
            # Update registry
            self.registry[version_id] = version_metadata
            
            # Update tags
            if version_tag:
                self._update_tag(version_tag, version_id)
            
            if set_as_latest:
                self._update_tag('latest', version_id)
            
            # Save registry
            self._save_registry()
            
            # Cleanup old versions
            if self.config.auto_cleanup_old_versions:
                self._cleanup_old_versions()
            
            # Audit log
            self._log_audit('version_saved', {
                'version_id': version_id,
                'tag': version_tag,
                'is_latest': set_as_latest
            })
            
            logger.info(
                "Model version saved: %s (tag=%s, latest=%s)",
                version_id, version_tag, set_as_latest
            )
            
            return version_id
            
        except Exception as e:
            logger.error("Failed to save version: %s", str(e), exc_info=True)
            raise VersioningError(f"Failed to save version: {e}") from e
    
    def rollback(
        self,
        checkpoint_path: Optional[str] = None,
        version_id: Optional[str] = None,
        version_tag: Optional[str] = None,
        validation_data: Optional[Tuple[Any, Any]] = None
    ) -> Any:
        """
        Rollback to a previous model version.
        
        Args:
            checkpoint_path: Direct path to model checkpoint (backward compatibility)
            version_id: Version ID to rollback to
            version_tag: Version tag to rollback to (e.g., "stable")
            validation_data: Optional (X, y) tuple for validation
        
        Returns:
            Loaded model
        
        Raises:
            RollbackError: If rollback fails
        """
        try:
            # Determine which version to load
            if checkpoint_path:
                # Backward compatibility
                logger.info("Loading model from checkpoint path (legacy mode)")
                model = joblib.load(checkpoint_path)
                target_version = "legacy"
            
            elif version_tag:
                # Load by tag
                target_version = self._resolve_tag(version_tag)
                if not target_version:
                    raise RollbackError(f"Version tag '{version_tag}' not found")
                
                version_info = self.registry[target_version]
                model = joblib.load(version_info['path'])
                logger.info("Loading model version by tag: %s -> %s", version_tag, target_version)
            
            elif version_id:
                # Load by version ID
                if version_id not in self.registry:
                    raise RollbackError(f"Version ID '{version_id}' not found")
                
                version_info = self.registry[version_id]
                model = joblib.load(version_info['path'])
                target_version = version_id
                logger.info("Loading model version by ID: %s", version_id)
            
            else:
                # Default to latest
                target_version = self._resolve_tag('latest')
                if not target_version:
                    raise RollbackError("No 'latest' version found")
                
                version_info = self.registry[target_version]
                model = joblib.load(version_info['path'])
                logger.info("Loading latest model version: %s", target_version)
            
            # Validate if configured
            if self.config.validate_after_rollback and validation_data is not None:
                self._validate_model(model, validation_data)
            
            # Audit log
            self._log_audit('rollback_performed', {
                'target_version': target_version,
                'version_tag': version_tag,
                'validated': validation_data is not None
            })
            
            logger.info("Rollback successful to version: %s", target_version)
            return model
            
        except Exception as e:
            logger.error("Rollback failed: %s", str(e), exc_info=True)
            raise RollbackError(f"Rollback failed: {e}") from e
    
    def list_versions(self, tag: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        List all available versions or filter by tag.
        
        Args:
            tag: Optional tag filter
        
        Returns:
            List of version metadata dictionaries
        """
        versions = list(self.registry.values())
        
        if tag:
            versions = [v for v in versions if v.get('tag') == tag]
        
        # Sort by timestamp (newest first)
        versions.sort(key=lambda v: v['timestamp'], reverse=True)
        
        return versions
    
    def get_version_info(self, version_id: str) -> Optional[Dict[str, Any]]:
        """Get metadata for a specific version."""
        return self.registry.get(version_id)
    
    def delete_version(self, version_id: str) -> None:
        """
        Delete a specific version.
        
        Args:
            version_id: Version ID to delete
        
        Raises:
            VersioningError: If deletion fails
        """
        if version_id not in self.registry:
            raise VersioningError(f"Version {version_id} not found")
        
        version_info = self.registry[version_id]
        
        # Don't delete if it's the latest or has important tags
        if version_info.get('tag') in ['stable', 'production']:
            logger.warning("Refusing to delete version with protected tag: %s", version_info['tag'])
            return
        
        # Delete files
        try:
            Path(version_info['path']).unlink(missing_ok=True)
            metadata_path = Path(version_info['path']).with_suffix('').as_posix() + '_metadata.json'
            Path(metadata_path).unlink(missing_ok=True)
        except Exception as e:
            logger.warning("Failed to delete version files: %s", str(e))
        
        # Remove from registry
        del self.registry[version_id]
        self._save_registry()
        
        # Audit log
        self._log_audit('version_deleted', {'version_id': version_id})
        
        logger.info("Version deleted: %s", version_id)
    
    def _cleanup_old_versions(self) -> None:
        """Automatically cleanup old versions beyond max_versions_to_keep."""
        versions = list(self.registry.values())
        
        # Sort by timestamp
        versions.sort(key=lambda v: v['timestamp'], reverse=True)
        
        # Keep protected tags and max_versions
        versions_to_delete = []
        kept_count = 0
        
        for version in versions:
            # Always keep protected tags
            if version.get('tag') in ['stable', 'production', 'latest']:
                continue
            
            kept_count += 1
            if kept_count > self.config.max_versions_to_keep:
                versions_to_delete.append(version['version_id'])
        
        # Delete old versions
        for version_id in versions_to_delete:
            try:
                self.delete_version(version_id)
            except Exception as e:
                logger.warning("Failed to cleanup version %s: %s", version_id, str(e))
        
        if versions_to_delete:
            logger.info("Cleaned up %d old versions", len(versions_to_delete))
    
    def _validate_model(self, model: Any, validation_data: Tuple[Any, Any]) -> None:
        """Validate model on validation data."""
        X_val, y_val = validation_data
        
        # Sample if too large
        if len(X_val) > self.config.validation_sample_size:
            indices = np.random.choice(len(X_val), self.config.validation_sample_size, replace=False)
            X_val = X_val.iloc[indices] if hasattr(X_val, 'iloc') else X_val[indices]
            y_val = y_val.iloc[indices] if hasattr(y_val, 'iloc') else y_val[indices]
        
        # Score model
        try:
            score = model.score(X_val, y_val)
            logger.info("Model validation score: %.4f", score)
            
            if score < 0.5:  # Basic sanity check
                logger.warning("Model validation score is suspiciously low: %.4f", score)
        
        except Exception as e:
            logger.error("Model validation failed: %s", str(e))
            raise RollbackError(f"Model validation failed: {e}") from e
    
    def _update_tag(self, tag: str, version_id: str) -> None:
        """Update a tag to point to a specific version."""
        # Remove tag from other versions
        for vid, info in self.registry.items():
            if info.get('tag') == tag and vid != version_id:
                info['tag'] = None
        
        # Set tag on target version
        if version_id in self.registry:
            self.registry[version_id]['tag'] = tag
    
    def _resolve_tag(self, tag: str) -> Optional[str]:
        """Resolve a tag to a version ID."""
        for version_id, info in self.registry.items():
            if info.get('tag') == tag:
                return version_id
        return None
    
    def _load_registry(self) -> Dict[str, Dict[str, Any]]:
        """Load version registry from disk."""
        if self.registry_path.exists():
            with open(self.registry_path, 'r') as f:
                return json.load(f)
        return {}
    
    def _save_registry(self) -> None:
        """Save version registry to disk."""
        with open(self.registry_path, 'w') as f:
            json.dump(self.registry, f, indent=2)
    
    def _load_audit_log(self) -> List[Dict[str, Any]]:
        """Load audit log from disk."""
        if self.audit_log_path.exists():
            with open(self.audit_log_path, 'r') as f:
                return json.load(f)
        return []
    
    def _save_audit_log(self) -> None:
        """Save audit log to disk."""
        with open(self.audit_log_path, 'w') as f:
            json.dump(self.audit_log, f, indent=2)
    
    def _log_audit(self, action: str, details: Dict[str, Any]) -> None:
        """Log an audit event."""
        event = {
            'timestamp': datetime.now().isoformat(),
            'action': action,
            'details': details
        }
        
        self.audit_log.append(event)
        self._save_audit_log()
        
        # Keep audit log manageable
        if len(self.audit_log) > 1000:
            self.audit_log = self.audit_log[-1000:]
            self._save_audit_log()
    
    def get_audit_log(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get recent audit events."""
        return self.audit_log[-limit:]
