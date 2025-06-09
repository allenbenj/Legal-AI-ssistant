from legal_ai_system.core.detailed_logging import get_detailed_logger, LogCategory

def main() -> None:
    logger = get_detailed_logger("LegalAISystem", LogCategory.SYSTEM)
    logger.info("Legal AI System initialized")

if __name__ == "__main__":
    main()
