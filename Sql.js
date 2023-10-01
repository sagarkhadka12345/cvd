import { Database } from "./Database.js";

class Sql {
  async start() {
    const response = await new Promise((resolve, reject) => {
      Database.getConnection((err, conn) => {
        if (err) reject(err);
        resolve(conn);
        console.log("Database connection successful");
      });
    });
    return response;
  }

  async $queryRaw(sql, parameterizedSQl) {
    let tempSQL = sql.split("?").length - 1;
    if (parameterizedSQl.length !== tempSQL)
      throw Error("ERROR IN LENGTH OF PARAMETER IN SQL");
    let response = await new Promise((resolve, reject) => {
      Database.getConnection((err, conn) => {
        if (err) reject(err);
        conn.query(sql, parameterizedSQl, (err, result) => {
          conn.release();
          if (err) reject(err);
          resolve(result);
        });
      });
    });
    return response;
  }
}
export default new Sql();
