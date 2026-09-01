// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

/**
 * @title AuditRegistry
 * @dev Privacy-Preserving Smart Contract Audit Registry for AI Financial Risk Outputs
 * Stores only cryptographic hash commitments and provenance metadata on-chain.
 * Financial details (amount, customer info, raw risk signals) remain strictly off-chain.
 */
contract AuditRegistry {
    struct AuditCommitment {
        string transactionId;
        string recordHash;
        string decision;
        string policyVersion;
        string modelVersion;
        uint256 timestamp;
        address submitter;
        bool exists;
    }

    // Mapping from transactionId => AuditCommitment
    mapping(string => AuditCommitment) private audits;
    
    // Array of transaction IDs for iteration/enumeration
    string[] private transactionIds;

    // Event emitted whenever a new audit hash commitment is recorded on-chain
    event AuditRecorded(
        string indexed transactionId,
        string recordHash,
        string decision,
        string policyVersion,
        string modelVersion,
        uint256 timestamp,
        address submitter
    );

    /**
     * @notice Records an audit commitment on the blockchain.
     * @param _transactionId Unique identifier of the transaction.
     * @param _decision Final decision (APPROVE, REJECT, HUMAN_APPROVED, HUMAN_REJECTED).
     * @param _policyVersion Version identifier of policy engine.
     * @param _modelVersion Version identifier of AI model.
     * @param _recordHash SHA-256 hash of the canonical off-chain audit record.
     */
    function recordAudit(
        string memory _transactionId,
        string memory _decision,
        string memory _policyVersion,
        string memory _modelVersion,
        string memory _recordHash
    ) external returns (bool) {
        require(bytes(_transactionId).length > 0, "Transaction ID cannot be empty");
        require(bytes(_recordHash).length > 0, "Record hash cannot be empty");

        if (!audits[_transactionId].exists) {
            transactionIds.push(_transactionId);
        }

        audits[_transactionId] = AuditCommitment({
            transactionId: _transactionId,
            recordHash: _recordHash,
            decision: _decision,
            policyVersion: _policyVersion,
            modelVersion: _modelVersion,
            timestamp: block.timestamp,
            submitter: msg.sender,
            exists: true
        });

        emit AuditRecorded(
            _transactionId,
            _recordHash,
            _decision,
            _policyVersion,
            _modelVersion,
            block.timestamp,
            msg.sender
        );

        return true;
    }

    /**
     * @notice Retrieves the recorded on-chain commitment for a given transaction ID.
     */
    function getAudit(string memory _transactionId)
        external
        view
        returns (
            string memory recordHash,
            string memory decision,
            string memory policyVersion,
            string memory modelVersion,
            uint256 timestamp,
            address submitter
        )
    {
        require(audits[_transactionId].exists, "Audit record does not exist");
        AuditCommitment memory item = audits[_transactionId];
        return (
            item.recordHash,
            item.decision,
            item.policyVersion,
            item.modelVersion,
            item.timestamp,
            item.submitter
        );
    }

    /**
     * @notice Checks whether an audit record exists for a transaction ID.
     */
    function hasAudit(string memory _transactionId) external view returns (bool) {
        return audits[_transactionId].exists;
    }

    /**
     * @notice Returns the total count of recorded audit commitments.
     */
    function getTotalAudits() external view returns (uint256) {
        return transactionIds.length;
    }

    /**
     * @notice Returns all stored transaction IDs.
     */
    function getAllTransactionIds() external view returns (string[] memory) {
        return transactionIds;
    }
}
