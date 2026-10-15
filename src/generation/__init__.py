"""Grounded Generation & Purchase Confidence Scoring: grounded generation and purchase confidence scoring."""

from .generator import ShopSmartGenerator
from .pcs import PCSConfig, calculate_pcs

__all__ = ["ShopSmartGenerator", "PCSConfig", "calculate_pcs"]
