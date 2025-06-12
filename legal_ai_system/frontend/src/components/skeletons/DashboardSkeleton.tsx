import React from 'react';
import CardSkeleton from './CardSkeleton';

const DashboardSkeleton: React.FC = () => (
  <div className="space-y-6">
    <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
      {Array.from({ length: 3 }).map((_, idx) => (
        <CardSkeleton key={idx} />
      ))}
    </div>
    <CardSkeleton />
    <CardSkeleton />
  </div>
);

export default DashboardSkeleton;
