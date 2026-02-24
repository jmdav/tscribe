import React from "react";
import ReactDOM from "react-dom/client";
import ClickableDiff from "./ClickableDiff";

const root = ReactDOM.createRoot(
  document.getElementById("root") as HTMLElement
);

root.render(
  <React.StrictMode>
    <ClickableDiff />
  </React.StrictMode>
);
