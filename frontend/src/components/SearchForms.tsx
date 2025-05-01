import React, { useState } from 'react';

interface SearchFormProps {
  onSearch: (query: string, isUrl: boolean, maxResults: number) => void;
}

const SearchForm: React.FC<SearchFormProps> = ({ onSearch }) => {
  const [query, setQuery] = useState('');
  const [isUrl, setIsUrl] = useState(false);
  const [maxResults, setMaxResults] = useState(5);
  
  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (query.trim()) {
      onSearch(query.trim(), isUrl, maxResults);
    }
  };

  return (
    <form onSubmit={handleSubmit} className="space-y-6 bg-white p-6 rounded-lg shadow-sm border border-gray-200">
      <div className="space-y-2">
        <label htmlFor="query" className="block text-sm font-medium text-gray-700">
          Job Description or Query
        </label>
        <div className="mt-1">
          <textarea
            id="query"
            name="query"
            rows={4}
            className="shadow-sm focus:ring-[#1a4a9e] focus:border-[#1a4a9e] block w-full sm:text-sm border-gray-300 rounded-md p-3"
            placeholder="I am hiring for Java developers who can also collaborate effectively with my business teams. Looking for an assessment(s) that can be completed in 40 minutes."
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            required
          />
        </div>
        <p className="text-sm text-gray-500">
          Enter your job requirements, skills needed, or paste a full job description.
        </p>
      </div>

      <div className="flex flex-col sm:flex-row space-y-4 sm:space-y-0 sm:space-x-6">
        <div className="flex items-center">
          <input
            id="isUrl"
            name="isUrl"
            type="checkbox"
            className="h-4 w-4 text-[#1a4a9e] focus:ring-[#1a4a9e] border-gray-300 rounded"
            checked={isUrl}
            onChange={(e) => setIsUrl(e.target.checked)}
          />
          <label htmlFor="isUrl" className="ml-2 block text-sm text-gray-700">
            This is a URL to a job description
          </label>
        </div>
        
        <div className="flex items-center space-x-2">
          <label htmlFor="maxResults" className="block text-sm text-gray-700">
            Max results:
          </label>
          <select
            id="maxResults"
            name="maxResults"
            className="block w-20 text-sm border-gray-300 rounded-md shadow-sm focus:ring-[#1a4a9e] focus:border-[#1a4a9e]"
            value={maxResults}
            onChange={(e) => setMaxResults(Number(e.target.value))}
          >
            {[1, 3, 5, 7, 10].map(num => (
              <option key={num} value={num}>{num}</option>
            ))}
          </select>
        </div>
      </div>

      <div>
        <button
          type="submit"
          className="w-full flex justify-center py-2 px-4 border border-transparent rounded-md shadow-sm text-sm font-medium text-white bg-[#1a4a9e] hover:bg-[#2a6ad2] focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-[#1a4a9e]"
        >
          Find Assessments
        </button>
      </div>
      
      <div className="pt-2">
        <div className="border-t border-gray-200">
          <h3 className="text-sm font-medium text-gray-500 pt-4 pb-2">Sample Queries</h3>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-2">
            {[
              "I am hiring for Java developers who can also collaborate effectively with my business teams. Looking for an assessment(s) that can be completed in 40 minutes.",
              "Looking to hire mid-level professionals who are proficient in Python, SQL and Java Script. Need an assessment package that can test all skills with max duration of 60 minutes.",
              "I need cognitive and personality tests for analyst roles, under 30 minutes each.",
              "Looking for leadership assessments suitable for senior management candidates."
            ].map((sample, idx) => (
              <button
                key={idx}
                type="button"
                className="text-left text-xs bg-gray-50 hover:bg-gray-100 p-2 rounded border border-gray-200 truncate"
                onClick={() => setQuery(sample)}
              >
                {sample.length > 70 ? `${sample.substring(0, 70)}...` : sample}
              </button>
            ))}
          </div>
        </div>
      </div>
    </form>
  );
};

export default SearchForm;