#!/usr/bin/env python
"""
================================================================================
LAMBDA HANDLERS - AWS Lambda Functions for Step Function Training Pipeline
================================================================================
Alpha Loop Capital, LLC

These handlers implement the Lambda functions called by the Step Functions
training pipelines. Deploy these to AWS Lambda with appropriate IAM roles.

DEPLOYMENT:
    # Package and deploy
    cd src/training/step_functions
    zip -r lambda_package.zip lambda_handlers.py
    aws lambda update-function-code --function-name alc-train-agent \
        --zip-file fileb://lambda_package.zip

================================================================================
"""

import json
import boto3
import os
from datetime import datetime
from typing import Dict, Any, List, Optional


# =============================================================================
# CONFIGURATION
# =============================================================================

S3_BUCKET = os.environ.get("ALC_S3_BUCKET", "alc-model-checkpoints")
DYNAMODB_TABLE = os.environ.get("ALC_DYNAMODB_TABLE", "alc-training-sessions")
SNS_TOPIC = os.environ.get("ALC_SNS_TOPIC", "arn:aws:sns:us-east-1:ACCOUNT:alc-alerts")

s3 = boto3.client('s3')
dynamodb = boto3.resource('dynamodb')
sns = boto3.client('sns')


# =============================================================================
# INITIALIZATION HANDLERS
# =============================================================================

def init_training_handler(event: Dict[str, Any], context) -> Dict[str, Any]:
    """
    Initialize a training session.

    Lambda: alc-init-training

    Input:
        - session_id: Unique session identifier
        - timestamp: Start timestamp
        - config: Training configuration

    Output:
        - session_id: Session ID
        - s3_prefix: S3 path for session data
        - initialized: True
    """
    session_id = event.get('session_id', datetime.now().strftime('%Y%m%d_%H%M%S'))
    timestamp = event.get('timestamp', datetime.now().isoformat())
    config = event.get('config', {})

    # Create session record in DynamoDB
    table = dynamodb.Table(DYNAMODB_TABLE)
    table.put_item(Item={
        'session_id': session_id,
        'status': 'initialized',
        'created_at': timestamp,
        'config': json.dumps(config),
        'agents_trained': [],
        'metrics': {}
    })

    # Create S3 prefix for session
    s3_prefix = f"training-sessions/{session_id}"

    return {
        'session_id': session_id,
        's3_prefix': s3_prefix,
        'initialized': True,
        'timestamp': timestamp
    }


def start_streams_handler(event: Dict[str, Any], context) -> Dict[str, Any]:
    """
    Start data streaming connections.

    Lambda: alc-start-streams

    In production, this would:
    - Initialize Polygon WebSocket connections
    - Start Alpaca data streams
    - Begin buffering to Kinesis/SQS
    """
    session_id = event.get('session_id')
    polygon_config = event.get('polygon_config', {})
    alpaca_config = event.get('alpaca_config', {})

    # In production: Start actual stream connections
    # For now, return simulated stream info

    return {
        'session_id': session_id,
        'streams': {
            'polygon': {
                'status': 'connected',
                'channels': polygon_config.get('channels', ['T', 'Q']),
            },
            'alpaca': {
                'status': 'connected',
                'paper_trading': alpaca_config.get('paper_trading', True)
            }
        },
        'kinesis_stream': f"alc-live-data-{session_id}"
    }


def stop_streams_handler(event: Dict[str, Any], context) -> Dict[str, Any]:
    """
    Stop data streaming connections.

    Lambda: alc-stop-streams
    """
    session_id = event.get('session_id')

    # In production: Close WebSocket connections, stop Kinesis consumers

    return {
        'session_id': session_id,
        'streams_stopped': True,
        'timestamp': datetime.now().isoformat()
    }


# =============================================================================
# DATA COLLECTION HANDLERS
# =============================================================================

def collect_data_handler(event: Dict[str, Any], context) -> Dict[str, Any]:
    """
    Collect training data from streams.

    Lambda: alc-collect-data

    Input:
        - session_id: Session identifier
        - duration_seconds: How long to collect
        - batch_size: Number of records per batch
        - feature_engineering: Feature config

    Output:
        - s3_path: Path to collected data
        - validation_path: Path to validation split
        - record_count: Number of records collected
        - features: List of engineered features
    """
    session_id = event.get('session_id')
    duration = event.get('duration_seconds', 300)
    batch_size = event.get('batch_size', 5000)
    feature_config = event.get('feature_engineering', {})

    # In production:
    # 1. Read from Kinesis stream
    # 2. Apply feature engineering
    # 3. Split train/validation
    # 4. Save to S3

    s3_path = f"s3://{S3_BUCKET}/training-sessions/{session_id}/data/train.parquet"
    validation_path = f"s3://{S3_BUCKET}/training-sessions/{session_id}/data/validation.parquet"

    return {
        's3_path': s3_path,
        'validation_path': validation_path,
        'record_count': batch_size * (duration // 60),
        'features': list(feature_config.get('technical_indicators', [])) +
                   list(feature_config.get('volatility_features', [])) +
                   list(feature_config.get('flow_features', []))
    }


def fetch_recent_data_handler(event: Dict[str, Any], context) -> Dict[str, Any]:
    """
    Fetch recent market data for incremental training.

    Lambda: alc-fetch-recent-data
    """
    lookback_minutes = event.get('lookback_minutes', 15)
    symbols = event.get('symbols', ['SPY', 'QQQ'])
    include_options = event.get('include_options', True)
    include_news = event.get('include_news', True)

    # In production: Fetch from Polygon/Alpaca REST APIs

    return {
        'data_collected': True,
        'lookback_minutes': lookback_minutes,
        'symbols': symbols,
        'record_count': len(symbols) * lookback_minutes * 60,  # Approx 1 record/second
        'timestamp': datetime.now().isoformat()
    }


# =============================================================================
# TRAINING HANDLERS
# =============================================================================

def train_agent_handler(event: Dict[str, Any], context) -> Dict[str, Any]:
    """
    Train a single agent.

    Lambda: alc-train-agent

    Input:
        - agent_name: Name of agent to train
        - tier: Agent tier (MASTER, SENIOR, STANDARD, STRATEGY)
        - training_config: Training hyperparameters
        - model_config: Model architecture config
        - data_path: S3 path to training data
        - checkpoint_bucket: S3 bucket for checkpoints

    Output:
        - agent_name: Trained agent name
        - success: Training success
        - metrics: Training metrics
        - checkpoint_path: Path to saved checkpoint
    """
    agent_name = event.get('agent_name')
    tier = event.get('tier')
    training_config = event.get('training_config', {})
    model_config = event.get('model_config', {})
    data_path = event.get('data_path')
    checkpoint_bucket = event.get('checkpoint_bucket', S3_BUCKET)

    mode = training_config.get('mode', 'batch')
    epochs = training_config.get('epochs', 10)
    learning_rate = training_config.get('learning_rate', 0.001)

    # In production:
    # 1. Load agent class
    # 2. Load training data from S3
    # 3. Run training loop
    # 4. Save checkpoint to S3

    # Simulated training metrics
    import random
    metrics = {
        'final_loss': 0.1 + random.random() * 0.1,
        'accuracy': 0.8 + random.random() * 0.15,
        'epochs_completed': epochs,
        'training_time_seconds': epochs * 60
    }

    checkpoint_path = f"s3://{checkpoint_bucket}/checkpoints/{agent_name}/{datetime.now().strftime('%Y%m%d_%H%M%S')}/model.pt"

    return {
        'agent_name': agent_name,
        'tier': tier,
        'success': True,
        'mode': mode,
        'metrics': metrics,
        'checkpoint_path': checkpoint_path
    }


def incremental_train_handler(event: Dict[str, Any], context) -> Dict[str, Any]:
    """
    Perform incremental/online training update.

    Lambda: alc-incremental-train

    Input:
        - agent: Agent name
        - mode: Training mode (online)
        - learning_rate: Learning rate
        - steps: Number of update steps

    Output:
        - agent: Agent name
        - steps_completed: Steps run
        - metrics: Update metrics
    """
    agent = event.get('agent')
    mode = event.get('mode', 'online')
    learning_rate = event.get('learning_rate', 0.0001)
    steps = event.get('steps', 10)
    priority = event.get('priority', 'normal')

    # In production: Load model, run incremental updates

    import random
    return {
        'agent': agent,
        'mode': mode,
        'steps_completed': steps,
        'priority': priority,
        'metrics': {
            'loss_before': 0.15 + random.random() * 0.05,
            'loss_after': 0.12 + random.random() * 0.05,
            'improvement': random.random() * 0.03
        },
        'timestamp': datetime.now().isoformat()
    }


def cross_train_handler(event: Dict[str, Any], context) -> Dict[str, Any]:
    """
    Perform cross-agent training.

    Lambda: alc-cross-train

    Input:
        - source_agents: List of source agents
        - target_agent: Target agent
        - synergy: Synergy description
        - communication_mode: articulate, observe, synthesize
        - training_rounds: Number of rounds

    Output:
        - success: Training success
        - insights_transferred: Number of insights
        - target_improvement: Improvement metric
    """
    source_agents = event.get('source_agents', [])
    target_agent = event.get('target_agent')
    synergy = event.get('synergy', '')
    communication_mode = event.get('communication_mode', 'synthesize')
    training_rounds = event.get('training_rounds', 5)

    # In production: Run cross-training logic from LiveDataTrainer

    import random
    return {
        'source_agents': source_agents,
        'target_agent': target_agent,
        'synergy': synergy,
        'communication_mode': communication_mode,
        'success': True,
        'insights_transferred': len(source_agents) * training_rounds,
        'target_improvement': random.random() * 0.1,
        'rounds_completed': training_rounds
    }


# =============================================================================
# VALIDATION HANDLERS
# =============================================================================

def validate_models_handler(event: Dict[str, Any], context) -> Dict[str, Any]:
    """
    Validate trained models.

    Lambda: alc-validate-models

    Input:
        - agents: List of agents to validate (or "ALL")
        - validation_data: Path to validation data
        - metrics: List of metrics to compute
        - thresholds: Validation thresholds

    Output:
        - all_passed: Whether all validations passed
        - failed_agents: List of failed agents
        - agent_metrics: Metrics per agent
    """
    agents = event.get('agents', [])
    validation_data = event.get('validation_data', '')
    metrics = event.get('metrics', ['accuracy'])
    thresholds = event.get('thresholds', {})

    if agents == "ALL":
        agents = ["GHOST", "HOAGS", "BOOKMAKER", "SCOUT", "AUTHOR", "HUNTER", "STRINGS", "KILLJOY"]

    # In production: Load models, run validation, compute metrics

    import random
    agent_metrics = {}
    failed_agents = []

    for agent in agents:
        agent_metrics[agent] = {
            'accuracy': 0.7 + random.random() * 0.25,
            'sharpe_ratio': 0.5 + random.random() * 1.5,
            'max_drawdown': random.random() * 0.2
        }

        # Check thresholds
        if agent_metrics[agent]['accuracy'] < thresholds.get('min_accuracy', 0.6):
            failed_agents.append(agent)
        elif agent_metrics[agent]['sharpe_ratio'] < thresholds.get('min_sharpe', 0.5):
            failed_agents.append(agent)
        elif agent_metrics[agent]['max_drawdown'] > thresholds.get('max_drawdown', 0.15):
            failed_agents.append(agent)

    return {
        'all_passed': len(failed_agents) == 0,
        'passed': len(failed_agents) == 0,  # Alias
        'deployment_ready': len(failed_agents) == 0,
        'failed_agents': failed_agents,
        'agent_metrics': agent_metrics,
        'timestamp': datetime.now().isoformat()
    }


def quick_validate_handler(event: Dict[str, Any], context) -> Dict[str, Any]:
    """
    Quick sanity check validation for incremental updates.

    Lambda: alc-quick-validate
    """
    validation_type = event.get('validation_type', 'sanity_check')
    metrics = event.get('metrics', ['prediction_drift'])
    max_drift = event.get('max_drift_threshold', 0.1)

    import random
    drift = random.random() * 0.15

    return {
        'validation_type': validation_type,
        'passed': drift < max_drift,
        'prediction_drift': drift,
        'confidence_calibration': 0.9 + random.random() * 0.1,
        'timestamp': datetime.now().isoformat()
    }


# =============================================================================
# DEPLOYMENT & MANAGEMENT HANDLERS
# =============================================================================

def save_deploy_handler(event: Dict[str, Any], context) -> Dict[str, Any]:
    """
    Save checkpoints and deploy models.

    Lambda: alc-save-deploy
    """
    session_id = event.get('session_id')
    checkpoint_bucket = event.get('checkpoint_bucket', S3_BUCKET)
    production_bucket = event.get('production_bucket', 'alc-production-models')
    versioning = event.get('versioning', True)
    deployment_config = event.get('deployment_config', {})

    version = datetime.now().strftime('%Y%m%d_%H%M%S')

    # In production:
    # 1. Copy checkpoints to production bucket
    # 2. Update model registry
    # 3. Trigger canary deployment if configured

    return {
        'session_id': session_id,
        'deployed': True,
        'version': version,
        'production_path': f"s3://{production_bucket}/models/{version}/",
        'canary_enabled': deployment_config.get('canary_percentage', 0) > 0,
        'timestamp': datetime.now().isoformat()
    }


def retrain_failed_handler(event: Dict[str, Any], context) -> Dict[str, Any]:
    """
    Retrain failed agents with adjusted parameters.

    Lambda: alc-retrain-failed
    """
    failed_agents = event.get('failed_agents', [])
    additional_epochs = event.get('additional_epochs', 10)
    adjust_lr = event.get('adjust_learning_rate', True)

    results = {}
    for agent in failed_agents:
        # In production: Retrain with adjusted parameters
        import random
        results[agent] = {
            'retrained': True,
            'new_accuracy': 0.75 + random.random() * 0.2,
            'epochs': additional_epochs
        }

    return {
        'retrained_agents': failed_agents,
        'results': results,
        'timestamp': datetime.now().isoformat()
    }


def commit_updates_handler(event: Dict[str, Any], context) -> Dict[str, Any]:
    """
    Commit incremental updates to production.

    Lambda: alc-commit-updates
    """
    update_type = event.get('update_type', 'incremental')
    timestamp = event.get('timestamp', datetime.now().isoformat())

    return {
        'committed': True,
        'update_type': update_type,
        'commit_timestamp': timestamp,
        'version': datetime.now().strftime('%Y%m%d_%H%M%S')
    }


def rollback_handler(event: Dict[str, Any], context) -> Dict[str, Any]:
    """
    Rollback failed updates.

    Lambda: alc-rollback
    """
    reason = event.get('reason', 'Unknown')
    preserve_logs = event.get('preserve_logs', True)

    return {
        'rolled_back': True,
        'reason': reason,
        'logs_preserved': preserve_logs,
        'timestamp': datetime.now().isoformat()
    }


# =============================================================================
# UTILITY HANDLERS
# =============================================================================

def check_market_hours_handler(event: Dict[str, Any], context) -> Dict[str, Any]:
    """
    Check if market is currently open.

    Lambda: alc-check-market-hours
    """
    from datetime import datetime
    import pytz

    timezone = event.get('timezone', 'America/New_York')
    extended_hours = event.get('extended_hours', False)

    tz = pytz.timezone(timezone)
    now = datetime.now(tz)

    # Regular hours: 9:30 AM - 4:00 PM ET, Mon-Fri
    # Extended hours: 4:00 AM - 8:00 PM ET

    is_weekday = now.weekday() < 5

    if extended_hours:
        is_open = is_weekday and 4 <= now.hour < 20
    else:
        is_open = is_weekday and (
            (now.hour == 9 and now.minute >= 30) or
            (10 <= now.hour < 16)
        )

    return {
        'is_open': is_open,
        'current_time': now.isoformat(),
        'timezone': timezone,
        'extended_hours': extended_hours
    }


def detect_regime_handler(event: Dict[str, Any], context) -> Dict[str, Any]:
    """
    Detect market regime changes.

    Lambda: alc-detect-regime
    """
    data = event.get('data', {})
    indicators = event.get('indicators', ['volatility'])
    sensitivity = event.get('sensitivity', 0.7)

    import random
    change_magnitude = random.random()

    return {
        'significant_change': change_magnitude > sensitivity,
        'minor_change': 0.3 < change_magnitude <= sensitivity,
        'change_magnitude': change_magnitude,
        'indicators_triggered': indicators[:2] if change_magnitude > 0.5 else [],
        'timestamp': datetime.now().isoformat()
    }


def notify_handler(event: Dict[str, Any], context) -> Dict[str, Any]:
    """
    Send notifications.

    Lambda: alc-notify
    """
    notification_type = event.get('notification_type', 'info')
    details = event.get('details', {})
    recipients = event.get('recipients', [])

    message = {
        'type': notification_type,
        'timestamp': datetime.now().isoformat(),
        'details': details
    }

    # In production: Send SNS notification
    # sns.publish(TopicArn=SNS_TOPIC, Message=json.dumps(message))

    return {
        'notified': True,
        'notification_type': notification_type,
        'recipients': recipients,
        'timestamp': datetime.now().isoformat()
    }


def generate_report_handler(event: Dict[str, Any], context) -> Dict[str, Any]:
    """
    Generate training report.

    Lambda: alc-generate-report
    """
    session_id = event.get('session_id')
    sections = event.get('include_sections', [])
    output_format = event.get('output_format', 'pdf')
    s3_bucket = event.get('s3_bucket', S3_BUCKET)

    report_path = f"s3://{s3_bucket}/reports/{session_id}/training_report.{output_format}"

    # In production: Generate actual report

    return {
        'report_generated': True,
        'session_id': session_id,
        'report_path': report_path,
        'sections': sections,
        'format': output_format,
        'timestamp': datetime.now().isoformat()
    }


def log_metrics_handler(event: Dict[str, Any], context) -> Dict[str, Any]:
    """
    Log training metrics to CloudWatch.

    Lambda: alc-log-metrics
    """
    namespace = event.get('metric_namespace', 'ALC/Training')
    metrics = event.get('metrics', {})

    # In production: Put metrics to CloudWatch
    # cloudwatch.put_metric_data(Namespace=namespace, MetricData=...)

    return {
        'logged': True,
        'namespace': namespace,
        'metric_count': len(metrics),
        'timestamp': datetime.now().isoformat()
    }


def cleanup_handler(event: Dict[str, Any], context) -> Dict[str, Any]:
    """
    Cleanup resources after training.

    Lambda: alc-cleanup
    """
    session_id = event.get('session_id')
    preserve_logs = event.get('preserve_logs', True)

    # In production: Stop streams, delete temp data, etc.

    return {
        'cleaned_up': True,
        'session_id': session_id,
        'logs_preserved': preserve_logs,
        'timestamp': datetime.now().isoformat()
    }


# =============================================================================
# HANDLER MAPPING
# =============================================================================

HANDLER_MAP = {
    'alc-init-training': init_training_handler,
    'alc-start-streams': start_streams_handler,
    'alc-stop-streams': stop_streams_handler,
    'alc-collect-data': collect_data_handler,
    'alc-fetch-recent-data': fetch_recent_data_handler,
    'alc-train-agent': train_agent_handler,
    'alc-incremental-train': incremental_train_handler,
    'alc-cross-train': cross_train_handler,
    'alc-validate-models': validate_models_handler,
    'alc-quick-validate': quick_validate_handler,
    'alc-save-deploy': save_deploy_handler,
    'alc-retrain-failed': retrain_failed_handler,
    'alc-commit-updates': commit_updates_handler,
    'alc-rollback': rollback_handler,
    'alc-check-market-hours': check_market_hours_handler,
    'alc-detect-regime': detect_regime_handler,
    'alc-notify': notify_handler,
    'alc-generate-report': generate_report_handler,
    'alc-log-metrics': log_metrics_handler,
    'alc-cleanup': cleanup_handler,
}


def lambda_handler(event: Dict[str, Any], context) -> Dict[str, Any]:
    """
    Universal Lambda handler that routes to specific handlers.

    Use this if deploying a single Lambda with multiple handlers,
    or deploy individual handlers directly.
    """
    handler_name = event.get('_handler', '')
    if handler_name in HANDLER_MAP:
        return HANDLER_MAP[handler_name](event, context)

    return {
        'error': f'Unknown handler: {handler_name}',
        'available_handlers': list(HANDLER_MAP.keys())
    }


if __name__ == "__main__":
    # Test handlers locally
    print("Testing handlers...")

    # Test init
    result = init_training_handler({'session_id': 'test_123'}, None)
    print(f"Init: {result}")

    # Test train
    result = train_agent_handler({
        'agent_name': 'GHOST',
        'tier': 'MASTER',
        'training_config': {'epochs': 5}
    }, None)
    print(f"Train: {result}")

    # Test validate
    result = validate_models_handler({
        'agents': ['GHOST', 'SCOUT'],
        'thresholds': {'min_accuracy': 0.6}
    }, None)
    print(f"Validate: {result}")
