import React from 'react';
import { spacing } from '../tokens';

export interface GridProps extends React.HTMLAttributes<HTMLDivElement> {
  columns?: number;
  gap?: keyof typeof spacing;
}

export const Grid: React.FC<GridProps> = ({
  columns = 1,
  gap = 'md',
  style,
  ...props
}) => {
  const baseStyle: React.CSSProperties = {
    display: 'grid',
    gridTemplateColumns: `repeat(${columns}, minmax(0, 1fr))`,
    gap: spacing[gap],
    ...style,
  };

  return <div {...props} style={baseStyle} />;
};

export default Grid;
