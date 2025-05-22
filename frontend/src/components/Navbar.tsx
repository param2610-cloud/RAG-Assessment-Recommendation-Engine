import React from 'react';

const Navbar: React.FC = () => {
  return (
    <nav className="bg-white border-b border-gray-200 shadow-sm">
      <div className="max-w-7xl mx-auto px-4 sm:px-6">
        <div className="flex justify-between h-16 items-center">
          <div className="flex items-center">
            <div className="flex-shrink-0">
              <img
                className="h-8 w-auto"
                src="/logo.png"
                alt="Logo"
                onError={(e) => {
                  const target = e.target as HTMLImageElement;
                  target.onerror = null;
                  target.style.display = 'none';
                }}
              />
              <span className="text-blue-600 font-bold text-xl ml-2">Assessment Recommender</span>
            </div>
          </div>
          <div className="hidden md:ml-6 md:flex md:items-center md:space-x-4">
            <a href="#" className="px-3 py-2 text-sm font-medium text-gray-600 hover:text-blue-600">
              Home
            </a>
            <a href="#" className="px-3 py-2 text-sm font-medium text-gray-600 hover:text-blue-600">
              About
            </a>
            <a href="#" className="px-3 py-2 text-sm font-medium text-gray-600 hover:text-blue-600">
              Features
            </a>
            <a href="#" className="px-3 py-2 text-sm font-medium text-gray-600 hover:bg-gray-100 hover:text-blue-600 rounded">
              Documentation
            </a>
          </div>
        </div>
      </div>
    </nav>
  );
};

export default Navbar;
