"""
Data Validation Module for TradingIA
Provides comprehensive data quality checks and validation for financial time series.
"""

import pandas as pd
from typing import Dict, List, Optional, Any
import logging
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

class ValidationSeverity(Enum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

@dataclass
class ValidationResult:
    """Result of a data validation check"""
    check_name: str
    severity: ValidationSeverity
    passed: bool
    message: str
    details: Optional[Dict[str, Any]] = None
    fix_suggestion: Optional[str] = None

class DataValidator:
    """
    Comprehensive data validator for financial time series data.
    Performs multiple validation checks to ensure data quality.
    """

    def __init__(self, strict_mode: bool = False):
        """
        Initialize validator

        Args:
            strict_mode: If True, warnings become errors
        """
        self.strict_mode = strict_mode
        self.results: List[ValidationResult] = []

    def validate_ohlc_relationships(self, df: pd.DataFrame) -> ValidationResult:
        """
        Validate OHLC relationships: O ≤ H, O ≤ C, L ≤ H, L ≤ C
        """
        if not all(col in df.columns for col in ['open', 'high', 'low', 'close']):
            return ValidationResult(
                "ohlc_relationships",
                ValidationSeverity.ERROR,
                False,
                "Missing OHLC columns",
                fix_suggestion="Ensure dataframe has 'open', 'high', 'low', 'close' columns"
            )

        # Check relationships
        invalid_o_h = (df['open'] > df['high']).sum()
        invalid_o_c = (df['open'] > df['close']).sum()
        invalid_l_h = (df['low'] > df['high']).sum()
        invalid_l_c = (df['low'] > df['close']).sum()

        total_invalid = invalid_o_h + invalid_o_c + invalid_l_h + invalid_l_c

        if total_invalid > 0:
            severity = ValidationSeverity.ERROR if self.strict_mode else ValidationSeverity.WARNING
            return ValidationResult(
                "ohlc_relationships",
                severity,
                False,
                f"Found {total_invalid} invalid OHLC relationships",
                {
                    "open > high": int(invalid_o_h),
                    "open > close": int(invalid_o_c),
                    "low > high": int(invalid_l_h),
                    "low > close": int(invalid_l_c)
                },
                "Fix OHLC data or remove invalid rows"
            )

        return ValidationResult(
            "ohlc_relationships",
            ValidationSeverity.INFO,
            True,
            "OHLC relationships are valid"
        )

    def detect_time_gaps(self, df: pd.DataFrame, expected_freq: str = '5min') -> ValidationResult:
        """
        Detect gaps in time series data
        """
        if 'timestamp' not in df.columns:
            return ValidationResult(
                "time_gaps",
                ValidationSeverity.ERROR,
                False,
                "Missing timestamp column",
                fix_suggestion="Ensure dataframe has 'timestamp' column"
            )

        # Sort by timestamp
        df_sorted = df.sort_values('timestamp').copy()
        df_sorted['timestamp'] = pd.to_datetime(df_sorted['timestamp'])

        # Calculate expected time differences
        expected_diff = pd.Timedelta(expected_freq)

        # Calculate actual differences
        time_diffs = df_sorted['timestamp'].diff()

        # Find gaps (more than expected)
        gaps = time_diffs > expected_diff
        gap_count = gaps.sum()

        if gap_count > 0:
            gap_details = []
            gap_indices = df_sorted.index[gaps]
            for idx in gap_indices:
                prev_time = df_sorted.loc[idx-1, 'timestamp'] if idx > 0 else None
                curr_time = df_sorted.loc[idx, 'timestamp']
                actual_gap = time_diffs.loc[idx]
                expected_gaps = int(actual_gap / expected_diff) - 1
                gap_details.append({
                    'index': idx,
                    'from_time': str(prev_time),
                    'to_time': str(curr_time),
                    'gap_duration': str(actual_gap),
                    'missing_bars': expected_gaps
                })

            severity = ValidationSeverity.WARNING if gap_count < len(df) * 0.1 else ValidationSeverity.ERROR
            return ValidationResult(
                "time_gaps",
                severity,
                False,
                f"Found {gap_count} time gaps in data",
                {'gaps': gap_details},
                "Consider filling gaps with interpolation or removing affected periods"
            )

        return ValidationResult(
            "time_gaps",
            ValidationSeverity.INFO,
            True,
            "No significant time gaps detected"
        )

    def check_duplicate_timestamps(self, df: pd.DataFrame) -> ValidationResult:
        """
        Check for duplicate timestamps
        """
        if 'timestamp' not in df.columns:
            return ValidationResult(
                "duplicate_timestamps",
                ValidationSeverity.ERROR,
                False,
                "Missing timestamp column"
            )

        duplicates = df['timestamp'].duplicated()
        dup_count = duplicates.sum()

        if dup_count > 0:
            dup_timestamps = df.loc[duplicates, 'timestamp'].unique()
            return ValidationResult(
                "duplicate_timestamps",
                ValidationSeverity.ERROR,
                False,
                f"Found {dup_count} duplicate timestamps",
                {
                    'duplicate_count': dup_count,
                    'duplicate_timestamps': list(dup_timestamps)
                },
                "Remove or aggregate duplicate timestamps"
            )

        return ValidationResult(
            "duplicate_timestamps",
            ValidationSeverity.INFO,
            True,
            "No duplicate timestamps found"
        )

    def validate_timezone(self, df: pd.DataFrame, expected_tz: str = 'UTC') -> ValidationResult:
        """
        Validate and normalize timezone information
        """
        if 'timestamp' not in df.columns:
            return ValidationResult(
                "timezone",
                ValidationSeverity.ERROR,
                False,
                "Missing timestamp column"
            )

        # Check if timestamps are timezone-aware
        tz_aware_count = 0
        for ts in df['timestamp']:
            if hasattr(ts, 'tz') and ts.tz is not None:
                tz_aware_count += 1

        if tz_aware_count == 0:
            # Convert to UTC
            df_copy = df.copy()
            df_copy['timestamp'] = pd.to_datetime(df_copy['timestamp']).dt.tz_localize(expected_tz)
            return ValidationResult(
                "timezone",
                ValidationSeverity.WARNING,
                False,
                "Timestamps are timezone-naive, converted to UTC",
                {'converted_to_utc': True},
                "Consider storing timestamps with timezone information"
            )

        # Check if all have the same timezone
        timezones = []
        for ts in df['timestamp']:
            if hasattr(ts, 'tz') and ts.tz is not None:
                tz_name = str(ts.tz)
                timezones.append(tz_name)

        unique_tz = list(set(timezones))

        if len(unique_tz) > 1:
            return ValidationResult(
                "timezone",
                ValidationSeverity.ERROR,
                False,
                f"Multiple timezones found: {unique_tz}",
                {'timezones': unique_tz},
                f"Normalize all timestamps to {expected_tz}"
            )

        if unique_tz and unique_tz[0] != expected_tz:
            return ValidationResult(
                "timezone",
                ValidationSeverity.WARNING,
                False,
                f"Timezone is {unique_tz[0]}, expected {expected_tz}",
                {'current_tz': unique_tz[0], 'expected_tz': expected_tz},
                f"Convert timezone to {expected_tz}"
            )

        return ValidationResult(
            "timezone",
            ValidationSeverity.INFO,
            True,
            f"All timestamps are in {expected_tz} timezone"
        )

    def detect_look_ahead_bias(self, df: pd.DataFrame) -> ValidationResult:
        """
        Detect potential look-ahead bias in the data
        """
        if 'timestamp' not in df.columns:
            return ValidationResult(
                "look_ahead_bias",
                ValidationSeverity.ERROR,
                False,
                "Missing timestamp column"
            )

        # Check if data is sorted chronologically
        is_sorted = df['timestamp'].is_monotonic_increasing
        if not is_sorted:
            return ValidationResult(
                "look_ahead_bias",
                ValidationSeverity.ERROR,
                False,
                "Data is not sorted chronologically",
                fix_suggestion="Sort data by timestamp ascending"
            )

        # Check for future timestamps
        now = pd.Timestamp.now(tz='UTC')
        future_count = 0
        for ts in df['timestamp']:
            if pd.Timestamp(ts) > now:
                future_count += 1

        if future_count > 0:
            return ValidationResult(
                "look_ahead_bias",
                ValidationSeverity.CRITICAL,
                False,
                f"Found {future_count} timestamps in the future",
                {'future_timestamps': future_count},
                "Remove future timestamps or check data source"
            )

        return ValidationResult(
            "look_ahead_bias",
            ValidationSeverity.INFO,
            True,
            "No look-ahead bias detected"
        )

    def validate_volume(self, df: pd.DataFrame) -> ValidationResult:
        """
        Validate volume data (> 0, reasonable ranges)
        """
        if 'volume' not in df.columns:
            return ValidationResult(
                "volume",
                ValidationSeverity.WARNING,
                True,
                "No volume column found (acceptable for some assets)"
            )

        # Check for negative volume
        negative_vol = (df['volume'] < 0).sum()
        if negative_vol > 0:
            return ValidationResult(
                "volume",
                ValidationSeverity.ERROR,
                False,
                f"Found {negative_vol} negative volume values",
                fix_suggestion="Remove or correct negative volume values"
            )

        # Check for zero volume
        zero_vol = (df['volume'] == 0).sum()
        zero_pct = zero_vol / len(df) * 100

        if zero_pct > 10:  # More than 10% zeros
            severity = ValidationSeverity.WARNING if zero_pct < 20 else ValidationSeverity.ERROR
            return ValidationResult(
                "volume",
                severity,
                False,
                f"Found {zero_vol} zero volume values ({zero_pct:.1f}%)",
                {'zero_volume_count': zero_vol, 'zero_percentage': zero_pct},
                "Investigate zero volume periods - may indicate data issues"
            )

        # Check for extreme volume outliers
        vol_q75 = df['volume'].quantile(0.75)
        vol_q25 = df['volume'].quantile(0.25)
        iqr = vol_q75 - vol_q25
        upper_bound = vol_q75 + 3 * iqr

        extreme_vol = (df['volume'] > upper_bound).sum()
        if extreme_vol > 0:
            return ValidationResult(
                "volume",
                ValidationSeverity.WARNING,
                False,
                f"Found {extreme_vol} extreme volume outliers",
                {'extreme_count': extreme_vol, 'upper_bound': upper_bound},
                "Review extreme volume values for data quality"
            )

        return ValidationResult(
            "volume",
            ValidationSeverity.INFO,
            True,
            "Volume data appears valid"
        )

    def check_large_dataset(self, df: pd.DataFrame, max_rows: int = 100000) -> ValidationResult:
        """
        Check for large datasets that may cause performance issues
        """
        row_count = len(df)

        if row_count > max_rows:
            return ValidationResult(
                "large_dataset",
                ValidationSeverity.WARNING,
                False,
                f"Dataset has {row_count:,} rows (recommended max: {max_rows:,})",
                {
                    'row_count': row_count,
                    'max_recommended': max_rows,
                    'excess_rows': row_count - max_rows
                },
                "Consider sampling or chunking large datasets for better performance"
            )

        return ValidationResult(
            "large_dataset",
            ValidationSeverity.INFO,
            True,
            f"Dataset size ({row_count:,} rows) is within recommended limits"
        )

    def run_all_validations(self, df: pd.DataFrame,
                          expected_freq: str = '5min',
                          expected_tz: str = 'UTC') -> List[ValidationResult]:
        """
        Run all validation checks
        """
        self.results = []

        # Run all checks
        checks = [
            lambda: self.validate_ohlc_relationships(df),
            lambda: self.detect_time_gaps(df, expected_freq),
            lambda: self.check_duplicate_timestamps(df),
            lambda: self.validate_timezone(df, expected_tz),
            lambda: self.detect_look_ahead_bias(df),
            lambda: self.validate_volume(df),
            lambda: self.check_large_dataset(df)
        ]

        for check in checks:
            try:
                result = check()
                self.results.append(result)
            except Exception as e:
                logger.error(f"Validation check failed: {e}")
                self.results.append(ValidationResult(
                    "validation_error",
                    ValidationSeverity.ERROR,
                    False,
                    f"Validation check failed: {str(e)}"
                ))

        return self.results

    def get_summary(self) -> Dict[str, Any]:
        """
        Get summary of validation results
        """
        if not self.results:
            return {"status": "no_validations_run"}

        total_checks = len(self.results)
        passed = sum(1 for r in self.results if r.passed)
        errors = sum(1 for r in self.results if r.severity == ValidationSeverity.ERROR)
        warnings = sum(1 for r in self.results if r.severity == ValidationSeverity.WARNING)
        critical = sum(1 for r in self.results if r.severity == ValidationSeverity.CRITICAL)

        status = "passed" if errors == 0 and critical == 0 else "failed"

        return {
            "status": status,
            "total_checks": total_checks,
            "passed": passed,
            "errors": errors,
            "warnings": warnings,
            "critical": critical,
            "results": [
                {
                    "check": r.check_name,
                    "severity": r.severity.value,
                    "passed": r.passed,
                    "message": r.message
                }
                for r in self.results
            ]
        }

def validate_dataframe(df: pd.DataFrame,
                      strict_mode: bool = False,
                      expected_freq: str = '5min',
                      expected_tz: str = 'UTC') -> Dict[str, Any]:
    """
    Convenience function to validate a dataframe
    """
    validator = DataValidator(strict_mode=strict_mode)
    validator.run_all_validations(df, expected_freq, expected_tz)
    return validator.get_summary()