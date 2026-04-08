/**
 * Adds two numbers.
 * @param {number} a - The first number.
 * @param {number} b - The second number.
 * @returns {number} The sum of a and b.
 */
const func=(a, b) => a + b;

/**
 * Test function that demonstrates the usage of the func function and MyClass.
 * It logs the result of func(2, 3) and creates an instance of MyClass to demonstrate its greet method.
 */
module.exports = ()=>{
    console.log(func(2, 3)); // Output: 5
}

/**
 * Test function that demonstrates the usage of the func function and MyClass.
 * It logs the result of func(2, 3) and creates an instance of MyClass to demonstrate its greet method.
 */
function testFunc() {
    console.log(func(2, 3)); // Output: 5
}

/**
 * A simple class that demonstrates a greeting method.
 */
class MyClass {
    constructor(name) {
        this.name = name;
    }

    greet() {
        return `Hello, ${this.name}!`;
    }
}

/**
 * Test function that demonstrates the usage of the func function and MyClass.
 * It logs the result of func(2, 3) and creates an instance of MyClass to demonstrate its greet method.
 */
async function testMyClass() {
    const myInstance = new MyClass("Alice");
    console.log(myInstance.greet()); // Output: Hello, Alice!
}
