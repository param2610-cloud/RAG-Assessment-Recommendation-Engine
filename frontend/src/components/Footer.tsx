import React from 'react';

const Footer: React.FC = () => {
  return (
    <footer className="bg-gray-100 pt-8 pb-6 border-t border-gray-200">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
          <div>
            <h3 className="text-sm font-semibold text-gray-600 uppercase tracking-wider">Product</h3>
            <ul className="mt-4 space-y-2">
              <li>
                <a href="#" className="text-sm text-gray-500 hover:text-blue-600">About</a>
              </li>
              <li>
                <a href="#" className="text-sm text-gray-500 hover:text-blue-600">Features</a>
              </li>
              <li>
                <a href="#" className="text-sm text-gray-500 hover:text-blue-600">Documentation</a>
              </li>
            </ul>
          </div>
          <div>
            <h3 className="text-sm font-semibold text-gray-600 uppercase tracking-wider">Resources</h3>
            <ul className="mt-4 space-y-2">
              <li>
                <a href="#" className="text-sm text-gray-500 hover:text-blue-600">Blog</a>
              </li>
              <li>
                <a href="#" className="text-sm text-gray-500 hover:text-blue-600">Help Center</a>
              </li>
              <li>
                <a href="#" className="text-sm text-gray-500 hover:text-blue-600">Community</a>
              </li>
            </ul>
          </div>
          <div>
            <h3 className="text-sm font-semibold text-gray-600 uppercase tracking-wider">Company</h3>
            <ul className="mt-4 space-y-2">
              <li>
                <a href="#" className="text-sm text-gray-500 hover:text-blue-600">About Us</a>
              </li>
              <li>
                <a href="#" className="text-sm text-gray-500 hover:text-blue-600">Careers</a>
              </li>
              <li>
                <a href="#" className="text-sm text-gray-500 hover:text-blue-600">Contact</a>
              </li>
            </ul>
          </div>
        </div>
        <div className="mt-8 border-t border-gray-200 pt-6 flex flex-col md:flex-row justify-between items-center">
          <p className="text-xs text-gray-500">&copy; {new Date().getFullYear()} Assessment Recommender. All rights reserved.</p>
          <div className="mt-4 md:mt-0 flex space-x-4">
            <a href="#" className="text-xs text-gray-500 hover:text-blue-600">Privacy Policy</a>
            <a href="#" className="text-xs text-gray-500 hover:text-blue-600">Terms of Service</a>
            <a href="#" className="text-xs text-gray-500 hover:text-blue-600">Contact</a>
          </div>
        </div>
      </div>
    </footer>
  );
};

export default Footer;