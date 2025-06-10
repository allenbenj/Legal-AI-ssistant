import React from 'react';
import { colors, spacing, typography, radii, shadows, animations } from '../tokens';

export type ButtonVariant = 'primary' | 'secondary' | 'outline';
export type ButtonSize = 'sm' | 'md' | 'lg';

export interface ButtonProps extends React.ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: ButtonVariant;
  size?: ButtonSize;
}

const sizeStyles: Record<ButtonSize, React.CSSProperties> = {
  sm: {
    fontSize: typography.fontSize.sm,
    padding: `${spacing.xs} ${spacing.sm}`,
    borderRadius: radii.sm,
  },
  md: {
    fontSize: typography.fontSize.md,
    padding: `${spacing.sm} ${spacing.md}`,
    borderRadius: radii.md,
  },
  lg: {
    fontSize: typography.fontSize.lg,
    padding: `${spacing.md} ${spacing.lg}`,
    borderRadius: radii.lg,
  },
};

const variantStyles: Record<ButtonVariant, React.CSSProperties> = {
  primary: {
    backgroundColor: colors.primary,
    color: colors.white,
    border: `1px solid ${colors.primary}`,
  },
  secondary: {
    backgroundColor: colors.gray200,
    color: colors.gray900,
    border: `1px solid ${colors.gray300}`,
  },
  outline: {
    backgroundColor: 'transparent',
    color: colors.primary,
    border: `1px solid ${colors.primary}`,
  },
};

export const Button: React.FC<ButtonProps> = ({
  variant = 'primary',
  size = 'md',
  style,
  ...props
}) => {
  const baseStyle: React.CSSProperties = {
    fontFamily: typography.fontFamily,
    fontWeight: typography.fontWeight.medium,
    cursor: 'pointer',
    transition: `background-color ${animations.fast}`,
    boxShadow: shadows.sm,
    ...sizeStyles[size],
    ...variantStyles[variant],
    ...style,
  };

  return <button {...props} style={baseStyle} />;
};

export default Button;
