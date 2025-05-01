import React from 'react';
import { Assessment } from '../types/Assessment';

interface ResultsTableProps {
  assessments: Assessment[];
}

const ResultsTable: React.FC<ResultsTableProps> = ({ assessments }) => {
  const getTestTypeLabels = (testTypes: string[] | undefined) => {
    if (!testTypes || !Array.isArray(testTypes) || testTypes.length === 0) {
      return 'N/A';
    }
    
    const types = {
      'C': 'Cognitive',
      'P': 'Personality',
      'A': 'Aptitude',
      'B': 'Behavioral',
      'K': 'Knowledge & Skills',
      'S': 'Simulation'
    };
    
    return testTypes.map(type => types[type as keyof typeof types] || type).join(', ');
  };

  return (
    <div className="overflow-x-auto shadow-sm border border-gray-200 rounded-lg">
      <table className="min-w-full divide-y divide-gray-200">
        <thead className="bg-gray-50">
          <tr>
            <th scope="col" className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
              Assessment
            </th>
            <th scope="col" className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
              Duration
            </th>
            <th scope="col" className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
              Test Type
            </th>
            <th scope="col" className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
              Remote Testing
            </th>
            <th scope="col" className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
              Adaptive/IRT
            </th>
          </tr>
        </thead>
        <tbody className="bg-white divide-y divide-gray-200">
          {assessments?.map((assessment) => (
            <tr key={assessment?.url || Math.random().toString()} className="hover:bg-gray-50 transition-colors">
              <td className="px-6 py-4">
                <div className="flex flex-col">
                  <a 
                    href={assessment?.url} 
                    target="_blank" 
                    rel="noopener noreferrer"
                    className="text-[#1a4a9e] hover:text-[#2a6ad2] font-medium"
                  >
                    {assessment?.name || 'Unnamed Assessment'}
                  </a>
                  <p className="text-sm text-gray-500 mt-1 line-clamp-2">{assessment?.description || 'No description available'}</p>
                  {assessment?.job_levels && assessment.job_levels.length > 0 && (
                    <div className="mt-2 flex flex-wrap gap-1">
                      {assessment.job_levels.map(level => (
                        <span key={level || Math.random().toString()} className="inline-flex items-center px-2 py-0.5 rounded text-xs font-medium bg-blue-50 text-[#1a4a9e] border border-blue-100">
                          {level}
                        </span>
                      ))}
                    </div>
                  )}
                </div>
              </td>
              <td className="px-6 py-4 whitespace-nowrap">
                <span className="px-2 py-1 text-sm font-medium bg-gray-100 rounded">
                  {assessment?.duration || 'N/A'} {assessment?.duration ? 'min' : ''}
                </span>
              </td>
              <td className="px-6 py-4 whitespace-nowrap">
                <div className="flex flex-col">
                  <span className="text-sm">{getTestTypeLabels(assessment?.test_types)}</span>
                  <span className="text-xs text-gray-500">{assessment?.test_types?.join(', ') || 'N/A'}</span>
                </div>
              </td>
              <td className="px-6 py-4 whitespace-nowrap text-center">
                {assessment?.remote_testing ? (
                  <span className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-green-100 text-green-800">
                    Yes
                  </span>
                ) : (
                  <span className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-red-100 text-red-800">
                    No
                  </span>
                )}
              </td>
              <td className="px-6 py-4 whitespace-nowrap text-center">
                {assessment?.adaptive_irt ? (
                  <span className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-green-100 text-green-800">
                    Yes
                  </span>
                ) : (
                  <span className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-red-100 text-red-800">
                    No
                  </span>
                )}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
};

export default ResultsTable;