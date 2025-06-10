import React from 'react';
import { radii, shadows, spacing } from '../tokens';

export interface CardProps extends React.HTMLAttributes<HTMLDivElement> {}

export const Card: React.FC<CardProps> = ({ style, ...props }) => {
  const baseStyle: React.CSSProperties = {
    borderRadius: radii.md,
    boxShadow: shadows.md,
    padding: spacing.md,
    backgroundColor: '#fff',
    ...style,
  };

  return <div {...props} style={baseStyle} />;
};

export default Card;
