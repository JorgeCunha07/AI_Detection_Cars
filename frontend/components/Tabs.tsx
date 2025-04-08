'use client'

import { useState } from 'react';

interface Tab {
  id: string;
  label: string;
}

interface TabsProps {
  tabs: Tab[];
  children: React.ReactNode[];
}

export default function Tabs({ tabs, children }: TabsProps) {
  const [activeTab, setActiveTab] = useState(tabs[0].id);

  return (
    <div>
      <div className="flex space-x-4 border-b">
        {tabs.map((tab) => (
          <button
            key={tab.id}
            onClick={() => setActiveTab(tab.id)}
            className={`py-2 px-4 text-sm font-medium ${activeTab === tab.id ? 'border-b-2 border-blue-500' : 'text-gray-500'}`}
          >
            {tab.label}
          </button>
        ))}
      </div>
      <div className="mt-4">
        {children.map((child, index) => (
          <div key={tabs[index].id} className={activeTab === tabs[index].id ? 'block' : 'hidden'}>
            {child}
          </div>
        ))}
      </div>
    </div>
  );
}
