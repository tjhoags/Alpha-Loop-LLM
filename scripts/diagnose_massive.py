from loguru import logger
from src.data_ingestion.sources.massive import massive_client

def diagnose():
    logger.info("Listing root folders in Massive S3...")
    
    # Try listing root
    try:
        files = massive_client.s3.list_objects_v2(Bucket="flatfiles", Delimiter="/")
        if "CommonPrefixes" in files:
            logger.info("Found Root Folders:")
            for p in files["CommonPrefixes"]:
                logger.info(f" - {p['Prefix']}")
        else:
            logger.warning("No folders found in root. Listing first 10 objects:")
            files = massive_client.s3.list_objects_v2(Bucket="flatfiles", MaxKeys=10)
            for obj in files.get("Contents", []):
                logger.info(f" - {obj['Key']}")
                
    except Exception as e:
        logger.error(f"Error listing bucket: {e}")

if __name__ == "__main__":
    diagnose()

