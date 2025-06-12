import React from 'react';
import { colors, spacing, typography, radii, shadows } from '../tokens';

export type AlertVariant = 'info' | 'success' | 'warning' | 'danger';

export interface AlertProps extends React.HTMLAttributes<HTMLDivElement> {
  variant?: AlertVariant;
}

const variantStyles: Record<AlertVariant, React.CSSProperties> = {
  info: { backgroundColor: colors.info, color: colors.white },
  success: { backgroundColor: colors.success, color: colors.white },
  warning: { backgroundColor: colors.warning, color: colors.black },
  danger: { backgroundColor: colors.danger, color: colors.white },
};

export const Alert: React.FC<AlertProps> = ({ variant = 'info', style, ...props }) => {
  const baseStyle: React.CSSProperties = {
    fontFamily: typography.fontFamily,
    padding: `${spacing.sm} ${spacing.md}`,
    borderRadius: radii.md,
    boxShadow: shadows.sm,
    ...variantStyles[variant],
    ...style,
  };

  return <div {...props} style={baseStyle} />;
};

export default Alert;
