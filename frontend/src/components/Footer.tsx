import React from 'react';

const Footer: React.FC = () => {
  return (
    <footer className="bg-gray-100 pt-8 pb-6 border-t border-gray-200">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
          <div>
            <h3 className="text-sm font-semibold text-gray-600 uppercase tracking-wider">Company</h3>
            <ul className="mt-4 space-y-2">
              <li>
                <a href="https://www.shl.com/about-shl/" className="text-sm text-gray-500 hover:text-[#1a4a9e]">About SHL</a>
              </li>
              <li>
                <a href="https://www.shl.com/solutions/" className="text-sm text-gray-500 hover:text-[#1a4a9e]">Solutions</a>
              </li>
              <li>
                <a href="https://www.shl.com/solutions/products/" className="text-sm text-gray-500 hover:text-[#1a4a9e]">Products</a>
              </li>
            </ul>
          </div>
          <div>
            <h3 className="text-sm font-semibold text-gray-600 uppercase tracking-wider">Client Resources</h3>
            <ul className="mt-4 space-y-2">
              <li>
                <a href="https://www.shl.com/solutions/products/product-catalog/" className="text-sm text-gray-500 hover:text-[#1a4a9e]">Product Catalog</a>
              </li>
              <li>
                <a href="https://www.shl.com/client-support/" className="text-sm text-gray-500 hover:text-[#1a4a9e]">Client Support</a>
              </li>
            </ul>
          </div>
          <div>
            <h3 className="text-sm font-semibold text-gray-600 uppercase tracking-wider">Candidate Resources</h3>
            <ul className="mt-4 space-y-2">
              <li>
                <a href="https://www.shl.com/candidate-support/" className="text-sm text-gray-500 hover:text-[#1a4a9e]">Candidate Support</a>
              </li>
              <li>
                <a href="https://www.shl.com/practice-tests/" className="text-sm text-gray-500 hover:text-[#1a4a9e]">Practice Tests</a>
              </li>
            </ul>
          </div>
        </div>
        <div className="mt-8 border-t border-gray-200 pt-6 flex flex-col md:flex-row justify-between items-center">
          <p className="text-xs text-gray-500">&copy; {new Date().getFullYear()} SHL and its affiliates. All rights reserved.</p>
          <div className="mt-4 md:mt-0 flex space-x-4">
            <a href="https://www.shl.com/privacy-notice/" className="text-xs text-gray-500 hover:text-[#1a4a9e]">Privacy Notice</a>
            <a href="https://www.shl.com/cookie-policy/" className="text-xs text-gray-500 hover:text-[#1a4a9e]">Cookie Policy</a>
            <a href="https://www.shl.com/legal-resources/" className="text-xs text-gray-500 hover:text-[#1a4a9e]">Legal Resources</a>
          </div>
        </div>
      </div>
    </footer>
  );
};

export default Footer;