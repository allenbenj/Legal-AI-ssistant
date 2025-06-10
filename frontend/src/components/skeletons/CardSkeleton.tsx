import React from 'react';
import Skeleton from './Skeleton';

const CardSkeleton: React.FC = () => (
  <div className="p-4 border rounded-lg shadow bg-white space-y-3">
    <Skeleton className="h-6 w-1/3" />
    <Skeleton className="h-4 w-full" />
    <Skeleton className="h-4 w-5/6" />
  </div>
);

export default CardSkeleton;
