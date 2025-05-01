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
                src="/shl-logo.png"
                alt="SHL Logo"
                onError={(e) => {
                  // Fallback if image doesn't load
                  const target = e.target as HTMLImageElement;
                  target.onerror = null;
                  target.style.display = 'none';
                }}
              />
              {/* Text logo fallback */}
              <span className="text-[#1a4a9e] font-bold text-xl ml-2">SHL Assessment Recommender</span>
            </div>
          </div>
          <div className="hidden md:ml-6 md:flex md:items-center md:space-x-4">
            <a href="https://www.shl.com" target="_blank" rel="noopener noreferrer" className="px-3 py-2 text-sm font-medium text-gray-600 hover:text-[#1a4a9e]">
              Home
            </a>
            <a href="https://www.shl.com/solutions/" target="_blank" rel="noopener noreferrer" className="px-3 py-2 text-sm font-medium text-gray-600 hover:text-[#1a4a9e]">
              Solutions
            </a>
            <a href="https://www.shl.com/solutions/products/" target="_blank" rel="noopener noreferrer" className="px-3 py-2 text-sm font-medium text-gray-600 hover:text-[#1a4a9e]">
              Products
            </a>
            <a href="https://www.shl.com/solutions/products/product-catalog/" target="_blank" rel="noopener noreferrer" className="px-3 py-2 text-sm font-medium text-gray-600 hover:bg-gray-100 hover:text-[#1a4a9e] rounded">
              Product Catalog
            </a>
          </div>
        </div>
      </div>
    </nav>
  );
};

export default Navbar;
