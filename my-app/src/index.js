// import React from "react";
// import ReactDOM from "react-dom";

// class App extends React.Component {
//   constructor(props) {
//     super(props);
//     this.state = {
//       attributes: [],
//       attribute: ""
//     };

//     this.handleSubmitCourse = this.handleSubmitCourse.bind(this);
//   }

//   handleSubmitCourse(event) {
//     alert("Your selected value is: " + this.state.attribute);
//     event.preventDefault();
//   }

//   handleChangeCourse = event => {
//     this.setState({ attribute: event.target.value });
//   };

//   getUnique(arr, comp) {
//     const unique = arr
//       //store the comparison values in array
//       .map(e => e[comp])

//       // store the keys of the unique objects
//       .map((e, i, final) => final.indexOf(e) === i && i)

//       // eliminate the dead keys & store unique objects
//       .filter(e => arr[e])

//       .map(e => arr[e]);

//     return unique;
//   }

//   componentDidMount() {
//     const attributes = require("./courses.json");
//     this.setState({ attributes: attributes });
//   }

//   render() {
//     const uniqueAttribute = this.getUnique(this.state.attributes, "tag");

//     const attributes = this.state.attributes;
//     const attribute = this.state.attribute;

//     const filterDropdown = attributes.filter(function(result) {
//       return result.tag === attribute;
//     });

//     return (
//       <div>

//         <form onSubmit={this.handleSubmitCourse}>
//           <br />
//           <br />
//           <label>
//             Select filter attribute to display data: 
//             <select
//               value={this.state.attribute}
//               onChange={this.handleChangeCourse}
//             >
//               {uniqueAttribute.map(attribute => (
//                 <option key={attribute.id} value={attribute.tag}>
//                   {attribute.tag}
//                 </option>
//               ))}
//             </select>
//           </label>
//           <input type="submit" value="Submit" />
//           <div>
//             {filterDropdown.map(attribute => (
//               <div key={attribute.id} style={{ margin: "10px" }}>
//                 {attribute.attribute}
//                 <br />
//               </div>
//             ))}
//           </div>
//         </form>
//       </div>
//     );
//   }
// }

// ReactDOM.render(<App />, document.querySelector("#root"));

import React from "react";
import ReactDOM from "react-dom";

class App extends React.Component {
  constructor(props) {
    super(props);
    this.state = {
      customers: [],
      filters: {},
      filteredCustomers: []
    };

    this.handleSubmit = this.handleSubmit.bind(this);
    this.handleChangeAttribute = this.handleChangeAttribute.bind(this);
    this.handleChangeValue = this.handleChangeValue.bind(this);
  }

  handleSubmit(event) {
    event.preventDefault();
    const filteredCustomers = this.state.customers.filter(customer =>
      Object.entries(this.state.filters).every(([key, value]) => {
        if (value === "") return true; // Skip empty filter values
        return customer[key] === value;
      })
    );
    this.setState({ filteredCustomers });
  }

  handleChangeAttribute(event) {
    const attributeName = event.target.value;
    this.setState(prevState => ({
      filters: {
        ...prevState.filters,
        [attributeName]: ""
      }
    }));
  }

  handleChangeValue(event, attributeName) {
    const attributeValue = event.target.value;
    this.setState(prevState => ({
      filters: {
        ...prevState.filters,
        [attributeName]: attributeValue
      }
    }));
  }

  componentDidMount() {
    const customers = require("./data.json");
    this.setState({ customers });
  }

  render() {
    const { filters, filteredCustomers } = this.state;

    // Get list of available attributes from the first customer object
    const attributeOptions = Object.keys(this.state.customers[0] || {});

    return (
      <div>
        <form onSubmit={this.handleSubmit}>
          <br />
          <br />
          <label>
            Select attribute to filter:
            <select onChange={this.handleChangeAttribute}>
              <option value="">-- Select Attribute --</option>
              {attributeOptions.map(attribute => (
                <option key={attribute} value={attribute}>
                  {attribute}
                </option>
              ))}
            </select>
          </label>
          <br />
          <br />
          {Object.entries(filters).map(([attributeName, attributeValue]) => (
            <div key={attributeName}>
              <label>
                {attributeName}:
                <input
                  type="text"
                  value={attributeValue}
                  onChange={event => this.handleChangeValue(event, attributeName)}
                />
              </label>
            </div>
          ))}
          <br />
          <input type="submit" value="Submit" />
        </form>
        <br />
        <div>
          {filteredCustomers.map((customer, index) => (
            <div key={index}>
              {Object.entries(customer).map(([key, value]) => (
                <div key={key}>
                  <strong>{key}: </strong>
                  {value}
                </div>
              ))}
              <hr />
            </div>
          ))}
        </div>
      </div>
    );
  }
}

ReactDOM.render(<App />, document.querySelector("#root"));
