import React from 'react';

interface SkeletonProps extends React.HTMLAttributes<HTMLDivElement> {
  className?: string;
  style?: React.CSSProperties;
}

const Skeleton: React.FC<SkeletonProps> = ({ className = '', style, ...props }) => {
  return (
    <div
      className={`bg-gray-300 rounded-md animate-pulse ${className}`.trim()}
      style={style}
      {...props}
    />
  );
};

export default Skeleton;
