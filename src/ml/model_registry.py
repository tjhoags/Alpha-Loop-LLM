"""
Machine Learning Model Registry

Managing ML models with versioning, A/B testing, and performance tracking.

Why Basic Approaches Fail:
- Model versioning chaos (which model is in production?)
- No systematic A/B testing of models
- Losing track of model performance over time
- Can't reproduce old model results
- No model governance or audit trail
- Model drift detection missing

Our Creative Philosophy:
- Centralized model registry (MLflow-style)
- Semantic versioning (major.minor.patch)
- A/B testing framework for model comparison
- Performance tracking and monitoring
- Model lineage and reproducibility
- Automatic model drift detection
- Champion/challenger framework
- Model metadata and documentation

Elite institutions use model registries:
- Two Sigma: Hundreds of models in production, systematic registry
- Renaissance: Decades of model versions preserved and tracked
- Citadel: Real-time model performance monitoring
- AQR: Factor model registry with full lineage

Author: Tom Hogan
Date: 2025-12-09
"""

import hashlib
import json
import logging
import pickle
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class ModelStage(Enum):
    """Model lifecycle stages"""

    DEVELOPMENT = "development"  # Being developed/trained
    STAGING = "staging"  # Ready for testing
    PRODUCTION = "production"  # Live in production
    ARCHIVED = "archived"  # Retired but preserved
    DEPRECATED = "deprecated"  # Obsolete, do not use


class ModelType(Enum):
    """Model types"""

    SUPERVISED_REGRESSION = "supervised_regression"
    SUPERVISED_CLASSIFICATION = "supervised_classification"
    UNSUPERVISED_CLUSTERING = "unsupervised_clustering"
    REINFORCEMENT_LEARNING = "reinforcement_learning"
    TIME_SERIES_FORECASTING = "time_series_forecasting"
    ENSEMBLE = "ensemble"
    DEEP_LEARNING = "deep_learning"
    GRADIENT_BOOSTING = "gradient_boosting"
    LINEAR_MODEL = "linear_model"


class ModelFramework(Enum):
    """ML frameworks"""

    SCIKIT_LEARN = "scikit_learn"
    TENSORFLOW = "tensorflow"
    PYTORCH = "pytorch"
    XGBOOST = "xgboost"
    LIGHTGBM = "lightgbm"
    CATBOOST = "catboost"
    STATSMODELS = "statsmodels"
    CUSTOM = "custom"


@dataclass
class ModelMetadata:
    """Comprehensive model metadata"""

    model_id: str
    name: str
    version: str  # Semantic version (e.g., "1.2.3")
    model_type: ModelType
    framework: ModelFramework
    stage: ModelStage
    created_at: datetime
    created_by: str  # "Tom Hogan" or agent name
    description: str

    # Training metadata
    training_data_hash: str  # MD5 hash of training data
    training_samples: int
    features: List[str]
    target: str
    hyperparameters: Dict[str, Any]

    # Performance metrics
    train_metrics: Dict[str, float]
    validation_metrics: Dict[str, float]
    test_metrics: Dict[str, float]

    # Deployment metadata
    production_start_date: Optional[datetime] = None
    production_end_date: Optional[datetime] = None
    prediction_count: int = 0
    avg_inference_time_ms: float = 0.0

    # Lineage
    parent_model_id: Optional[str] = None  # If based on previous model
    related_models: List[str] = field(default_factory=list)

    # Tags and categorization
    tags: List[str] = field(default_factory=list)
    strategy: Optional[str] = None  # Which strategy uses this model

    # Additional metadata
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ModelVersion:
    """Single model version with serialized model"""

    metadata: ModelMetadata
    model_artifact: Any  # Serialized model (pickle, joblib, etc.)
    model_path: Path  # File path to saved model
    checksum: str  # Model file checksum for integrity
    size_bytes: int  # Model file size


@dataclass
class ABTestResult:
    """A/B test comparison result"""

    champion_id: str
    challenger_id: str
    metric: str  # Metric being compared (e.g., "sharpe_ratio")
    champion_score: float
    challenger_score: float
    improvement: float  # % improvement (positive = challenger better)
    statistical_significance: float  # p-value
    sample_size: int
    test_duration_days: int
    recommendation: str  # "promote", "keep_champion", "inconclusive"
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class ModelDriftDetection:
    """Model drift detection result"""

    model_id: str
    drift_detected: bool
    drift_score: float  # 0-1 (0 = no drift, 1 = severe drift)
    feature_drift: Dict[str, float]  # Feature → drift score
    prediction_drift: float  # Drift in prediction distribution
    performance_degradation: float  # % decline in performance
    recommendation: str  # "retrain", "monitor", "ok"
    timestamp: datetime = field(default_factory=datetime.now)


class ModelRegistry:
    """
    Centralized ML model registry.

    Features:
    - Model versioning and lifecycle management
    - A/B testing framework
    - Performance tracking
    - Drift detection
    - Model lineage and reproducibility
    """

    def __init__(
        self,
        registry_path: Path,
        auto_save: bool = True,
    ):
        self.registry_path = Path(registry_path)
        self.registry_path.mkdir(parents=True, exist_ok=True)

        self.models_path = self.registry_path / "models"
        self.models_path.mkdir(exist_ok=True)

        self.metadata_path = self.registry_path / "metadata"
        self.metadata_path.mkdir(exist_ok=True)

        self.auto_save = auto_save

        # In-memory registry
        self.models: Dict[str, List[ModelVersion]] = {}  # model_name → [versions]
        self.champion_models: Dict[str, str] = {}  # model_name → version (production)

        # Load existing registry
        self._load_registry()

        logger.info(f"ModelRegistry initialized at {registry_path}")

    def register_model(
        self,
        name: str,
        model: Any,
        model_type: ModelType,
        framework: ModelFramework,
        version: Optional[str] = None,
        description: str = "",
        features: Optional[List[str]] = None,
        target: Optional[str] = None,
        hyperparameters: Optional[Dict[str, Any]] = None,
        train_metrics: Optional[Dict[str, float]] = None,
        validation_metrics: Optional[Dict[str, float]] = None,
        test_metrics: Optional[Dict[str, float]] = None,
        tags: Optional[List[str]] = None,
        strategy: Optional[str] = None,
    ) -> ModelMetadata:
        """
        Register a new model version.

        Args:
            name: Model name (e.g., "momentum_predictor")
            model: Trained model object
            model_type: Type of model
            framework: ML framework used
            version: Semantic version (auto-increment if None)
            description: Model description
            features: List of feature names
            target: Target variable name
            hyperparameters: Model hyperparameters
            train_metrics: Training metrics
            validation_metrics: Validation metrics
            test_metrics: Test metrics
            tags: Model tags
            strategy: Strategy name that uses this model

        Returns:
            ModelMetadata for registered model
        """
        # Auto-increment version if not provided
        if version is None:
            version = self._get_next_version(name)

        # Generate model ID
        model_id = f"{name}_v{version}"

        # Create metadata
        metadata = ModelMetadata(
            model_id=model_id,
            name=name,
            version=version,
            model_type=model_type,
            framework=framework,
            stage=ModelStage.DEVELOPMENT,
            created_at=datetime.now(),
            created_by="Tom Hogan",
            description=description,
            training_data_hash="",  # TODO: Calculate from data
            training_samples=0,
            features=features or [],
            target=target or "",
            hyperparameters=hyperparameters or {},
            train_metrics=train_metrics or {},
            validation_metrics=validation_metrics or {},
            test_metrics=test_metrics or {},
            tags=tags or [],
            strategy=strategy,
        )

        # Save model artifact
        model_path = self.models_path / f"{model_id}.pkl"
        with open(model_path, "wb") as f:
            pickle.dump(model, f)

        # Calculate checksum
        checksum = self._calculate_checksum(model_path)
        size_bytes = model_path.stat().st_size

        # Create model version
        model_version = ModelVersion(
            metadata=metadata,
            model_artifact=model,
            model_path=model_path,
            checksum=checksum,
            size_bytes=size_bytes,
        )

        # Add to registry
        if name not in self.models:
            self.models[name] = []
        self.models[name].append(model_version)

        # Save metadata
        self._save_metadata(metadata)

        if self.auto_save:
            self._save_registry()

        logger.info(f"Registered model {model_id} ({size_bytes:,} bytes)")
        return metadata

    def promote_to_production(
        self,
        name: str,
        version: str,
    ) -> bool:
        """
        Promote a model version to production.

        This makes it the "champion" model for this name.
        """
        model_version = self._get_model_version(name, version)
        if model_version is None:
            logger.error(f"Model {name} v{version} not found")
            return False

        # Update stage
        model_version.metadata.stage = ModelStage.PRODUCTION
        model_version.metadata.production_start_date = datetime.now()

        # Demote previous champion (if any)
        if name in self.champion_models:
            old_champion_version = self.champion_models[name]
            old_champion = self._get_model_version(name, old_champion_version)
            if old_champion:
                old_champion.metadata.stage = ModelStage.ARCHIVED
                old_champion.metadata.production_end_date = datetime.now()

        # Set new champion
        self.champion_models[name] = version

        # Save
        self._save_metadata(model_version.metadata)
        if self.auto_save:
            self._save_registry()

        logger.info(f"Promoted {name} v{version} to production")
        return True

    def get_production_model(self, name: str) -> Optional[Any]:
        """Get the current production model"""
        if name not in self.champion_models:
            logger.warning(f"No production model for {name}")
            return None

        version = self.champion_models[name]
        model_version = self._get_model_version(name, version)

        if model_version is None:
            return None

        return model_version.model_artifact

    def run_ab_test(
        self,
        champion_name: str,
        champion_version: str,
        challenger_name: str,
        challenger_version: str,
        test_data: pd.DataFrame,
        test_labels: pd.Series,
        metric_func: Callable,
        metric_name: str = "accuracy",
        test_duration_days: int = 30,
    ) -> ABTestResult:
        """
        Run A/B test between champion and challenger models.

        Args:
            champion_name: Champion model name
            champion_version: Champion version
            challenger_name: Challenger model name
            challenger_version: Challenger version
            test_data: Test dataset
            test_labels: Test labels
            metric_func: Function to calculate metric (higher = better)
            metric_name: Metric name
            test_duration_days: Test duration in days

        Returns:
            ABTestResult with comparison
        """
        # Load models
        champion = self._get_model_version(champion_name, champion_version)
        challenger = self._get_model_version(challenger_name, challenger_version)

        if champion is None or challenger is None:
            raise ValueError("Champion or challenger model not found")

        # Make predictions
        champion_preds = champion.model_artifact.predict(test_data)
        challenger_preds = challenger.model_artifact.predict(test_data)

        # Calculate metrics
        champion_score = metric_func(test_labels, champion_preds)
        challenger_score = metric_func(test_labels, challenger_preds)

        # Calculate improvement
        improvement = (challenger_score - champion_score) / champion_score if champion_score != 0 else 0.0

        # Statistical significance (paired t-test)
        # Simplified - assumes binary classification or regression residuals
        if hasattr(champion_preds, '__len__'):
            champion_errors = np.abs(test_labels - champion_preds)
            challenger_errors = np.abs(test_labels - challenger_preds)

            from scipy import stats
            t_stat, p_value = stats.ttest_rel(champion_errors, challenger_errors)
            statistical_significance = p_value
        else:
            statistical_significance = 1.0  # Inconclusive

        # Recommendation
        if improvement > 0.05 and statistical_significance < 0.05:
            recommendation = "promote"  # Challenger is significantly better
        elif improvement < -0.05 and statistical_significance < 0.05:
            recommendation = "keep_champion"  # Champion is better
        else:
            recommendation = "inconclusive"  # No clear winner

        result = ABTestResult(
            champion_id=f"{champion_name}_v{champion_version}",
            challenger_id=f"{challenger_name}_v{challenger_version}",
            metric=metric_name,
            champion_score=float(champion_score),
            challenger_score=float(challenger_score),
            improvement=float(improvement),
            statistical_significance=float(statistical_significance),
            sample_size=len(test_data),
            test_duration_days=test_duration_days,
            recommendation=recommendation,
        )

        logger.info(
            f"A/B Test: {result.champion_id} vs {result.challenger_id} - "
            f"Improvement: {improvement:+.2%}, p-value: {statistical_significance:.3f}, "
            f"Recommendation: {recommendation}"
        )

        return result

    def detect_drift(
        self,
        name: str,
        version: str,
        new_data: pd.DataFrame,
        new_labels: Optional[pd.Series] = None,
        reference_data: Optional[pd.DataFrame] = None,
    ) -> ModelDriftDetection:
        """
        Detect model drift (data drift and performance drift).

        Args:
            name: Model name
            version: Model version
            new_data: New production data
            new_labels: New labels (if available)
            reference_data: Reference data (training data)

        Returns:
            ModelDriftDetection result
        """
        model_version = self._get_model_version(name, version)
        if model_version is None:
            raise ValueError(f"Model {name} v{version} not found")

        # Feature drift (distribution shift using KL divergence or KS test)
        feature_drift = {}
        if reference_data is not None:
            for col in new_data.columns:
                if col in reference_data.columns:
                    # Kolmogorov-Smirnov test
                    from scipy import stats
                    ks_stat, p_value = stats.ks_2samp(
                        reference_data[col].dropna(),
                        new_data[col].dropna()
                    )
                    feature_drift[col] = float(ks_stat)  # 0-1 (1 = max drift)

        # Overall feature drift (average)
        avg_feature_drift = np.mean(list(feature_drift.values())) if feature_drift else 0.0

        # Prediction drift (distribution of predictions)
        predictions = model_version.model_artifact.predict(new_data)

        # Simplified: check if prediction distribution changed
        # In production, compare to historical prediction distribution
        prediction_mean = np.mean(predictions)
        prediction_std = np.std(predictions)

        # Assume reference prediction stats (would come from metadata in production)
        reference_pred_mean = 0.5  # Placeholder
        reference_pred_std = 0.2  # Placeholder

        prediction_drift = abs(prediction_mean - reference_pred_mean) / reference_pred_std if reference_pred_std > 0 else 0.0

        # Performance drift (if labels available)
        performance_degradation = 0.0
        if new_labels is not None:
            # Calculate current performance
            from sklearn.metrics import accuracy_score
            current_performance = accuracy_score(new_labels, predictions)

            # Compare to validation performance
            reference_performance = model_version.metadata.validation_metrics.get("accuracy", current_performance)
            performance_degradation = (reference_performance - current_performance) / reference_performance if reference_performance > 0 else 0.0

        # Overall drift score (weighted combination)
        drift_score = (
            0.4 * avg_feature_drift +
            0.3 * min(1.0, prediction_drift) +
            0.3 * min(1.0, performance_degradation)
        )

        # Detection and recommendation
        drift_detected = drift_score > 0.3  # Threshold

        if drift_score > 0.5 or performance_degradation > 0.1:
            recommendation = "retrain"
        elif drift_score > 0.3:
            recommendation = "monitor"
        else:
            recommendation = "ok"

        result = ModelDriftDetection(
            model_id=f"{name}_v{version}",
            drift_detected=drift_detected,
            drift_score=float(drift_score),
            feature_drift=feature_drift,
            prediction_drift=float(prediction_drift),
            performance_degradation=float(performance_degradation),
            recommendation=recommendation,
        )

        logger.info(
            f"Drift Detection for {name} v{version}: "
            f"Drift Score: {drift_score:.2f}, "
            f"Performance Degradation: {performance_degradation:+.2%}, "
            f"Recommendation: {recommendation}"
        )

        return result

    def list_models(
        self,
        stage: Optional[ModelStage] = None,
        model_type: Optional[ModelType] = None,
        tags: Optional[List[str]] = None,
    ) -> List[ModelMetadata]:
        """List models with optional filters"""
        all_models = []

        for name, versions in self.models.items():
            for version in versions:
                metadata = version.metadata

                # Apply filters
                if stage and metadata.stage != stage:
                    continue
                if model_type and metadata.model_type != model_type:
                    continue
                if tags and not any(tag in metadata.tags for tag in tags):
                    continue

                all_models.append(metadata)

        return all_models

    def _get_model_version(self, name: str, version: str) -> Optional[ModelVersion]:
        """Get specific model version"""
        if name not in self.models:
            return None

        for model_version in self.models[name]:
            if model_version.metadata.version == version:
                return model_version

        return None

    def _get_next_version(self, name: str) -> str:
        """Auto-increment version"""
        if name not in self.models or len(self.models[name]) == 0:
            return "1.0.0"

        # Get latest version
        versions = [v.metadata.version for v in self.models[name]]
        latest = max(versions)

        # Increment minor version
        major, minor, patch = latest.split(".")
        next_minor = int(minor) + 1
        return f"{major}.{next_minor}.0"

    def _calculate_checksum(self, file_path: Path) -> str:
        """Calculate MD5 checksum of file"""
        hash_md5 = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()

    def _save_metadata(self, metadata: ModelMetadata):
        """Save model metadata to JSON"""
        metadata_file = self.metadata_path / f"{metadata.model_id}.json"

        # Convert to dict
        metadata_dict = {
            "model_id": metadata.model_id,
            "name": metadata.name,
            "version": metadata.version,
            "model_type": metadata.model_type.value,
            "framework": metadata.framework.value,
            "stage": metadata.stage.value,
            "created_at": metadata.created_at.isoformat(),
            "created_by": metadata.created_by,
            "description": metadata.description,
            "features": metadata.features,
            "target": metadata.target,
            "hyperparameters": metadata.hyperparameters,
            "train_metrics": metadata.train_metrics,
            "validation_metrics": metadata.validation_metrics,
            "test_metrics": metadata.test_metrics,
            "tags": metadata.tags,
            "strategy": metadata.strategy,
        }

        with open(metadata_file, "w") as f:
            json.dump(metadata_dict, f, indent=2)

    def _save_registry(self):
        """Save registry state"""
        registry_file = self.registry_path / "registry.json"

        registry_data = {
            "champion_models": self.champion_models,
            "updated_at": datetime.now().isoformat(),
        }

        with open(registry_file, "w") as f:
            json.dump(registry_data, f, indent=2)

    def _load_registry(self):
        """Load registry state"""
        registry_file = self.registry_path / "registry.json"

        if not registry_file.exists():
            return

        with open(registry_file, "r") as f:
            registry_data = json.load(f)

        self.champion_models = registry_data.get("champion_models", {})

        # Load all model metadata
        for metadata_file in self.metadata_path.glob("*.json"):
            with open(metadata_file, "r") as f:
                metadata_dict = json.load(f)

            # Reconstruct metadata (simplified - full version would load models too)
            # This is just for listing purposes
            pass


# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score
    from sklearn.model_selection import train_test_split

    # Sample data
    np.random.seed(42)
    X = np.random.randn(1000, 10)
    y = (X[:, 0] + X[:, 1] > 0).astype(int)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

    # Initialize registry
    registry = ModelRegistry(registry_path=Path("./model_registry"))

    # Train model
    model = RandomForestClassifier(n_estimators=100, max_depth=10)
    model.fit(X_train, y_train)

    # Evaluate
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    train_acc = accuracy_score(y_train, y_pred_train)
    test_acc = accuracy_score(y_test, y_pred_test)

    # Register model
    metadata = registry.register_model(
        name="momentum_predictor",
        model=model,
        model_type=ModelType.SUPERVISED_CLASSIFICATION,
        framework=ModelFramework.SCIKIT_LEARN,
        description="Random Forest momentum prediction model",
        features=[f"feature_{i}" for i in range(10)],
        target="momentum_signal",
        hyperparameters={"n_estimators": 100, "max_depth": 10},
        train_metrics={"accuracy": train_acc},
        test_metrics={"accuracy": test_acc},
        tags=["momentum", "random_forest"],
        strategy="MomentumAgent",
    )

    print("\n=== Model Registered ===")
    print(f"Model ID: {metadata.model_id}")
    print(f"Version: {metadata.version}")
    print(f"Train Accuracy: {train_acc:.2%}")
    print(f"Test Accuracy: {test_acc:.2%}")

    # Promote to production
    registry.promote_to_production("momentum_predictor", metadata.version)
    print(f"\n✅ Promoted to production")

    # Get production model
    prod_model = registry.get_production_model("momentum_predictor")
    print(f"Production model loaded: {prod_model is not None}")

    # List all models
    all_models = registry.list_models(stage=ModelStage.PRODUCTION)
    print(f"\nProduction models: {len(all_models)}")
    for m in all_models:
        print(f"  - {m.model_id}: {m.description}")

    print("\n✅ ML Model Registry - Tom Hogan")
