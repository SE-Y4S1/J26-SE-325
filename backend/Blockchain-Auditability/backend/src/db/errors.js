/** Shared by both audit stores, so the controller's instanceof check holds for either. */
class DuplicateTransactionError extends Error {
  constructor(message) {
    super(message);
    this.name = "DuplicateTransactionError";
    this.statusCode = 409;
  }
}

module.exports = { DuplicateTransactionError };
