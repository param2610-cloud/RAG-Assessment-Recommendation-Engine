import React, { useState } from 'react';
import './App.css';
import { Assessment } from './types/Assessment';
import SearchForm from './components/SearchForms';
import Navbar from './components/Navbar';
import LoadingSpinner from './components/LoadingSpinner';
import ErrorMessage from './components/ErrorMessage';
import ResultsTable from './components/ResultTable';
import Footer from './components/Footer';

const App: React.FC = () => {
  const [results, setResults] = useState<Assessment[]>([]);
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const [searchQuery, setSearchQuery] = useState<string>('');

  const handleSearch = async (query: string, isUrl: boolean, maxResults: number) => {
    setLoading(true);
    setError(null);
    setSearchQuery(query);
    try {
      
      const response = await fetch(`${import.meta.env.VITE_BACKEND_DOMAIN}/search?query=${encodeURIComponent(query)}&is_url=${isUrl}&max_results=${maxResults}`);
      
      if (!response.ok) {
        throw new Error(`API error: ${response.status}`);
      }
      
      const data = await response.json();
      console.log(data);
      setResults(data.results);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'An unknown error occurred');
      setResults([]);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="flex flex-col min-h-screen bg-gray-50">
      <Navbar />
      
      <main className="flex-grow">
        <div className="bg-[#f6f6f6] py-8 border-b border-gray-200">
          <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
            <h1 className="text-3xl font-bold text-[#1a4a9e]">SHL Assessment Recommendation System</h1>
            <p className="mt-2 text-lg text-gray-600">
              Find the perfect assessments to match your hiring needs
            </p>
          </div>
        </div>
        
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
          <div className="mb-8">
            <h2 className="text-xl font-medium text-gray-800 mb-4">Enter Your Requirements</h2>
            <SearchForm onSearch={handleSearch} />
          </div>
          
          {loading ? (
            <LoadingSpinner />
          ) : error ? (
            <ErrorMessage message={error} />
          ) : results.length > 0 ? (
            <div className="bg-white p-6 rounded-lg shadow-sm border border-gray-200">
              <h2 className="text-xl font-semibold text-gray-800 mb-4">
                Recommended Assessments
              </h2>
              <p className="text-sm text-gray-500 mb-6 italic">
                Based on: "{searchQuery}"
              </p>
              <ResultsTable assessments={results} />
            </div>
          ) : searchQuery ? (
            <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6 text-center">
              <p className="text-gray-600">No assessments found matching your criteria.</p>
            </div>
          ) : null}
          
          <div className="mt-12 mb-8">
            <div className="bg-white p-6 rounded-lg shadow-sm border border-gray-200">
              <h2 className="text-lg font-semibold text-[#1a4a9e] mb-4">Why Choose SHL Assessments?</h2>
              <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mt-6">
                <div className="flex flex-col items-center text-center p-4">
                  <div className="bg-blue-50 rounded-full p-3 mb-4">
                    <svg className="h-8 w-8 text-[#1a4a9e]" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M9 12l2 2 4-4m5.618-4.016A11.955 11.955 0 0112 2.944a11.955 11.955 0 01-8.618 3.04A12.02 12.02 0 003 9c0 5.591 3.824 10.29 9 11.622 5.176-1.332 9-6.03 9-11.622 0-1.042-.133-2.052-.382-3.016z"></path>
                    </svg>
                  </div>
                  <h3 className="text-md font-medium text-gray-900">Proven Reliability</h3>
                  <p className="mt-2 text-sm text-gray-500">Science-backed assessments with high reliability and validity.</p>
                </div>
                <div className="flex flex-col items-center text-center p-4">
                  <div className="bg-blue-50 rounded-full p-3 mb-4">
                    <svg className="h-8 w-8 text-[#1a4a9e]" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M3 6l3 1m0 0l-3 9a5.002 5.002 0 006.001 0M6 7l3 9M6 7l6-2m6 2l3-1m-3 1l-3 9a5.002 5.002 0 006.001 0M18 7l3 9m-3-9l-6-2m0-2v2m0 16V5m0 16H9m3 0h3"></path>
                    </svg>
                  </div>
                  <h3 className="text-md font-medium text-gray-900">Fair & Unbiased</h3>
                  <p className="mt-2 text-sm text-gray-500">Designed to minimize bias and provide equal opportunity to all candidates.</p>
                </div>
                <div className="flex flex-col items-center text-center p-4">
                  <div className="bg-blue-50 rounded-full p-3 mb-4">
                    <svg className="h-8 w-8 text-[#1a4a9e]" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M21 12a9 9 0 01-9 9m9-9a9 9 0 00-9-9m9 9H3m9 9a9 9 0 01-9-9m9 9c1.657 0 3-4.03 3-9s-1.343-9-3-9m0 18c-1.657 0-3-4.03-3-9s1.343-9 3-9m-9 9a9 9 0 019-9"></path>
                    </svg>
                  </div>
                  <h3 className="text-md font-medium text-gray-900">Global Reach</h3>
                  <p className="mt-2 text-sm text-gray-500">Available in multiple languages and adaptable to diverse cultural contexts.</p>
                </div>
              </div>
            </div>
          </div>
        </div>
      </main>
      
      <Footer />
    </div>
  );
};

export default App;