import { createPool } from "mysql2";

export const Database = createPool({
  connectionLimit: 20,
  host: "localhost",
  port: 3306,
  user: "root",
  password: "",
  database: "cvd",
  multipleStatements: true,
});
