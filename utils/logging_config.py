"""
Centralized Logging Configuration for Trading System

Usage:
    from utils.logging_config import get_logger, setup_logging
    
    # Setup once at application start
    setup_logging(log_level="INFO", log_to_file=True)
    
    # Get logger in any module
    logger = get_logger(__name__)
    logger.info("Message")
"""

import logging
import logging.handlers
import sys
import re
from pathlib import Path
from datetime import datetime
from typing import Optional

# Sensitive data patterns to filter
SENSITIVE_PATTERNS = [
    (r'api[_-]?key["\']?\s*[:=]\s*["\']?[\w-]+', 'api_key=***REDACTED***'),
    (r'secret[_-]?key["\']?\s*[:=]\s*["\']?[\w-]+', 'secret_key=***REDACTED***'),
    (r'password["\']?\s*[:=]\s*["\']?[^\s,}"\']+', 'password=***REDACTED***'),
    (r'token["\']?\s*[:=]\s*["\']?[\w-]+', 'token=***REDACTED***'),
    (r'APCA[_-]API[_-](KEY|SECRET)[_-]ID\s*[:=]\s*[\w-]+', 'ALPACA_CREDENTIAL=***REDACTED***'),
    (r'Bearer\s+[\w.-]+', 'Bearer ***REDACTED***'),
]


class SensitiveDataFilter(logging.Filter):
    """Filter to redact sensitive information from logs"""
    
    def filter(self, record: logging.LogRecord) -> bool:
        if hasattr(record, 'msg') and record.msg:
            msg = str(record.msg)
            for pattern, replacement in SENSITIVE_PATTERNS:
                msg = re.sub(pattern, replacement, msg, flags=re.IGNORECASE)
            record.msg = msg
        
        if hasattr(record, 'args') and record.args:
            new_args = []
            for arg in record.args:
                arg_str = str(arg)
                for pattern, replacement in SENSITIVE_PATTERNS:
                    arg_str = re.sub(pattern, replacement, arg_str, flags=re.IGNORECASE)
                new_args.append(arg_str)
            record.args = tuple(new_args)
        
        return True


class ColoredFormatter(logging.Formatter):
    """Colored formatter for console output"""
    
    COLORS = {
        'DEBUG': '\033[36m',     # Cyan
        'INFO': '\033[32m',      # Green
        'WARNING': '\033[33m',   # Yellow
        'ERROR': '\033[31m',     # Red
        'CRITICAL': '\033[35m',  # Magenta
    }
    RESET = '\033[0m'
    
    def format(self, record: logging.LogRecord) -> str:
        color = self.COLORS.get(record.levelname, self.RESET)
        record.levelname = f"{color}{record.levelname}{self.RESET}"
        return super().format(record)


def get_project_root() -> Path:
    """Get project root directory"""
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / 'pyproject.toml').exists():
            return parent
    return Path.cwd()


def setup_logging(
    log_level: str = "INFO",
    log_to_file: bool = True,
    log_dir: Optional[Path] = None,
    log_filename: Optional[str] = None,
    max_bytes: int = 10 * 1024 * 1024,  # 10MB
    backup_count: int = 5,
    colored_console: bool = True,
) -> logging.Logger:
    """
    Setup centralized logging configuration
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_to_file: Whether to log to file
        log_dir: Directory for log files (default: data/logs)
        log_filename: Custom log filename (default: trading_YYYYMMDD.log)
        max_bytes: Max size per log file before rotation
        backup_count: Number of backup files to keep
        colored_console: Use colored output in console
        
    Returns:
        Root logger configured
    """
    # Get or create root logger for trading system
    root_logger = logging.getLogger('trading')
    root_logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))
    
    # Clear existing handlers
    root_logger.handlers.clear()
    
    # Add sensitive data filter
    sensitive_filter = SensitiveDataFilter()
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.DEBUG)
    console_handler.addFilter(sensitive_filter)
    
    if colored_console and sys.stdout.isatty():
        console_format = ColoredFormatter(
            '%(asctime)s | %(levelname)s | %(name)s | %(message)s',
            datefmt='%H:%M:%S'
        )
    else:
        console_format = logging.Formatter(
            '%(asctime)s | %(levelname)s | %(name)s | %(message)s',
            datefmt='%H:%M:%S'
        )
    console_handler.setFormatter(console_format)
    root_logger.addHandler(console_handler)
    
    # File handler (if enabled)
    if log_to_file:
        if log_dir is None:
            log_dir = get_project_root() / 'data' / 'logs'
        log_dir.mkdir(parents=True, exist_ok=True)
        
        if log_filename is None:
            log_filename = f"trading_{datetime.now().strftime('%Y%m%d')}.log"
        
        file_handler = logging.handlers.RotatingFileHandler(
            log_dir / log_filename,
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding='utf-8'
        )
        file_handler.setLevel(logging.DEBUG)
        file_handler.addFilter(sensitive_filter)
        file_format = logging.Formatter(
            '%(asctime)s | %(levelname)s | %(name)s | %(filename)s:%(lineno)d | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(file_format)
        root_logger.addHandler(file_handler)
    
    # Suppress noisy libraries
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    logging.getLogger('requests').setLevel(logging.WARNING)
    logging.getLogger('alpaca').setLevel(logging.WARNING)
    logging.getLogger('matplotlib').setLevel(logging.WARNING)
    logging.getLogger('PIL').setLevel(logging.WARNING)
    
    root_logger.info(f"Logging initialized at {log_level} level")
    
    return root_logger


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger for a specific module
    
    Args:
        name: Module name (usually __name__)
        
    Returns:
        Logger instance
    """
    # Prefix with 'trading.' for hierarchy
    if not name.startswith('trading.'):
        name = f"trading.{name}"
    return logging.getLogger(name)


# Performance logging context manager
class LogPerformance:
    """Context manager for logging performance of operations"""
    
    def __init__(self, logger: logging.Logger, operation: str, level: int = logging.DEBUG):
        self.logger = logger
        self.operation = operation
        self.level = level
        self.start_time: Optional[float] = None
    
    def __enter__(self):
        import time
        self.start_time = time.perf_counter()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        import time
        duration = time.perf_counter() - self.start_time
        if exc_type:
            self.logger.error(f"{self.operation} failed after {duration:.3f}s: {exc_val}")
        else:
            self.logger.log(self.level, f"{self.operation} completed in {duration:.3f}s")
        return False


# Example usage / test
if __name__ == "__main__":
    # Setup logging
    setup_logging(log_level="DEBUG", log_to_file=False)
    
    # Get logger
    logger = get_logger(__name__)
    
    # Test different levels
    logger.debug("Debug message")
    logger.info("Info message")
    logger.warning("Warning message")
    logger.error("Error message")
    
    # Test sensitive data filtering
    logger.info("Connecting with api_key=sk-12345678")
    logger.info("Using token: Bearer abc123xyz")
    
    # Test performance logging
    import time
    with LogPerformance(logger, "Test operation", logging.INFO):
        time.sleep(0.1)
