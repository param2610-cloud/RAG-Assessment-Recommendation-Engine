import React from 'react';

const LoadingSpinner: React.FC = () => {
  return (
    <div className="flex flex-col justify-center items-center py-16 bg-white rounded-lg border border-gray-200 shadow-sm">
      <div className="animate-spin rounded-full h-12 w-12 border-t-2 border-b-2 border-[#1a4a9e]"></div>
      <span className="mt-4 text-gray-600 font-medium">Searching for the best assessments...</span>
      <p className="mt-2 text-sm text-gray-500">This may take a moment as we analyze your requirements.</p>
    </div>
  );
};

export default LoadingSpinner;