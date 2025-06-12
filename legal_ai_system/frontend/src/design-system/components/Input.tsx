import React from 'react';
import { colors, spacing, typography, radii, shadows } from '../tokens';

export interface InputProps extends React.InputHTMLAttributes<HTMLInputElement> {}

export const Input: React.FC<InputProps> = ({ style, ...props }) => {
  const baseStyle: React.CSSProperties = {
    fontFamily: typography.fontFamily,
    fontSize: typography.fontSize.md,
    padding: `${spacing.sm} ${spacing.md}`,
    border: `1px solid ${colors.gray300}`,
    borderRadius: radii.md,
    boxShadow: shadows.sm,
    width: '100%',
    ...style,
  };

  return <input {...props} style={baseStyle} />;
};

export default Input;
