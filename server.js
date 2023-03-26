const express = require("express");
const app = express();

const PORT = process.env.PORT || 8080;

app.use(express.static("views"));
app.use(express.urlencoded({ extended: true }));
app.use(express.json());

app.get("/", (req, res) => res.sendFile(path.join(__dirname, "/views/index.html")));

app.listen(PORT, () => {
  console.log("Listening on port " + PORT);
});
