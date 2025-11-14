"""
Configuración de tema minimalista para el dashboard.
"""

# Color palette
COLORS = {
    'primary': '#0F172A',        # Slate-900
    'secondary': '#64748B',      # Slate-500
    'success': '#10B981',        # Emerald-500
    'warning': '#F59E0B',        # Amber-500
    'danger': '#EF4444',         # Red-500
    'info': '#3B82F6',           # Blue-500

    'bg_main': '#FFFFFF',        # White
    'bg_card': '#F8FAFC',        # Slate-50
    'bg_hover': '#F1F5F9',       # Slate-100
    'bg_sidebar': '#F8FAFC',     # Slate-50

    'text_primary': '#0F172A',   # Slate-900
    'text_secondary': '#64748B', # Slate-500
    'text_tertiary': '#94A3B8',  # Slate-400
    'text_light': '#E2E8F0',     # Slate-200

    'border_default': '#E2E8F0', # Slate-200
    'border_hover': '#CBD5E1',   # Slate-300
}

# Typography
FONTS = {
    'family_primary': 'Inter, -apple-system, BlinkMacSystemFont, "Segoe UI"',
    'family_mono': '"IBM Plex Mono", monospace',

    'size_xs': '12px',
    'size_sm': '14px',
    'size_base': '16px',
    'size_lg': '18px',
    'size_xl': '20px',
    'size_2xl': '24px',
    'size_3xl': '30px',

    'weight_normal': 400,
    'weight_medium': 500,
    'weight_semibold': 600,
    'weight_bold': 700,
}

# Shadows
SHADOWS = {
    'sm': '0 1px 2px 0 rgba(0, 0, 0, 0.05)',
    'base': '0 1px 3px 0 rgba(0, 0, 0, 0.1), 0 1px 2px 0 rgba(0, 0, 0, 0.06)',
    'md': '0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06)',
    'lg': '0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05)',
    'xl': '0 20px 25px -5px rgba(0, 0, 0, 0.1), 0 10px 10px -5px rgba(0, 0, 0, 0.04)',
}

# Spacing
SPACING = {
    'xs': '4px',
    'sm': '8px',
    'md': '12px',
    'lg': '16px',
    'xl': '24px',
    'xxl': '32px',
}

# Border radius
RADIUS = {
    'sm': '4px',
    'base': '6px',
    'md': '8px',
    'lg': '12px',
    'xl': '16px',
}

# Transitions
TRANSITIONS = {
    'fast': '150ms ease-in-out',
    'base': '300ms ease-in-out',
    'slow': '500ms ease-in-out',
}