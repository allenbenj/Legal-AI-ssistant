import React from 'react';

export interface LoaderStage {
  label: string;
  completed: boolean;
}

interface ProgressiveLoaderProps {
  stages: LoaderStage[];
}

const ProgressiveLoader: React.FC<ProgressiveLoaderProps> = ({ stages }) => {
  return (
    <div className="space-y-1">
      {stages.map((stage, idx) => (
        <div key={idx} className="flex items-center gap-2 text-sm">
          <div
            className={
              stage.completed
                ? 'w-3 h-3 bg-green-500 rounded-full'
                : 'w-3 h-3 bg-gray-400 rounded-full animate-pulse'
            }
          />
          <span className={stage.completed ? 'text-gray-700' : 'text-gray-500'}>
            {stage.label}
          </span>
        </div>
      ))}
    </div>
  );
};

export default ProgressiveLoader;
