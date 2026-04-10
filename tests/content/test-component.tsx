import React, { useState } from 'react';

interface ButtonProps {
    label: string;
    onClick: () => void;
    variant?: 'primary' | 'secondary';
}
/**
 * Button component with Tailwind CSS styling.
 * @param param0 
 * @returns 
 */
const Button: React.FC<ButtonProps> = ({ label, onClick, variant = 'primary' }) => {
    const baseStyles = 'px-4 py-2 rounded font-medium transition-colors';
    const variantStyles = {
        primary: 'bg-blue-600 text-white hover:bg-blue-700',
        secondary: 'bg-gray-300 text-gray-900 hover:bg-gray-400',
    };

    return (
        <button className={`${baseStyles} ${variantStyles[variant]}`} onClick={onClick}>
            {label}
        </button>
    );
};
/**
 * TEST COMPONENT
 */
const TestComponent: React.FC = () => {
    const [count, setCount] = useState(0);

    const handleIncrement = () => setCount(count + 1);
    const handleReset = () => setCount(0);

    return (
        <div className="flex flex-col items-center justify-center h-screen bg-gray-100">
            <div className="bg-white p-8 rounded-lg shadow-lg">
                <h1 className="text-3xl font-bold mb-4 text-gray-800">React Counter</h1>
                <p className="text-xl text-gray-600 mb-6">Count: {count}</p>
                <div className="flex gap-4">
                    <Button label="Increment" onClick={handleIncrement} variant="primary" />
                    <Button label="Reset" onClick={handleReset} variant="secondary" />
                </div>
            </div>
        </div>
    );
};

export default TestComponent;