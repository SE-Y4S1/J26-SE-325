// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import "@openzeppelin/contracts/access/AccessControl.sol";

/**
 * @title AuditRegistry
 * @dev Privacy-Preserving Smart Contract Audit Registry for AI Risk Outputs
 * Anchors only cryptographic bytes32 commitments on-chain.
 * All sensitive financial data (amounts, customer info, risk scores, decisions) remain strictly off-chain.
 */
contract AuditRegistry is AccessControl {
    bytes32 public constant SUBMITTER_ROLE = keccak256("SUBMITTER_ROLE");

    struct AuditCommitment {
        bytes32 recordHash;
        bytes32 policyHash;
        uint256 timestamp;
        address submitter;
        bool exists;
    }

    // Mapping from pseudonymous auditId => AuditCommitment
    mapping(bytes32 => AuditCommitment) private audits;

    // Policy registry: policyHash => active boolean
    mapping(bytes32 => bool) private activePolicies;

    // Custom errors for gas efficiency and clear diagnostics
    error InvalidAuditId();
    error InvalidRecordHash();
    error PolicyNotActive();
    error AuditAlreadyExists();

    // Event emitted upon recording an audit commitment
    event AuditRecorded(
        bytes32 indexed auditId,
        bytes32 indexed policyHash,
        bytes32 recordHash,
        uint256 timestamp,
        address submitter
    );

    // Event emitted when a policy hash is registered or status changed
    event PolicyRegistered(bytes32 indexed policyHash, bool active);

    /**
     * @notice Constructor initializes contract with deployer as DEFAULT_ADMIN_ROLE and SUBMITTER_ROLE.
     */
    constructor() {
        _grantRole(DEFAULT_ADMIN_ROLE, msg.sender);
        _grantRole(SUBMITTER_ROLE, msg.sender);
    }

    /**
     * @notice Registers or activates a policy definition hash. Admin only.
     * @param _policyHash SHA-256 hash of the canonical policy specification.
     */
    function registerPolicy(bytes32 _policyHash) external onlyRole(DEFAULT_ADMIN_ROLE) {
        if (_policyHash == bytes32(0)) revert InvalidRecordHash();
        activePolicies[_policyHash] = true;
        emit PolicyRegistered(_policyHash, true);
    }

    /**
     * @notice Deactivates a policy hash if deprecated. Admin only.
     * @param _policyHash SHA-256 hash of the policy specification.
     */
    function deactivatePolicy(bytes32 _policyHash) external onlyRole(DEFAULT_ADMIN_ROLE) {
        activePolicies[_policyHash] = false;
        emit PolicyRegistered(_policyHash, false);
    }

    /**
     * @notice Records an immutable audit commitment. Submitter role only. Write-once: cannot overwrite or delete.
     * @param _auditId Pseudonymous unique commitment identifier (e.g. sha256(transactionId + ":" + salt)).
     * @param _recordHash SHA-256 hash of the canonical off-chain audit record.
     * @param _policyHash SHA-256 hash of the evaluated policy specification.
     */
    function recordAudit(
        bytes32 _auditId,
        bytes32 _recordHash,
        bytes32 _policyHash
    ) external onlyRole(SUBMITTER_ROLE) returns (bool) {
        if (_auditId == bytes32(0)) revert InvalidAuditId();
        if (_recordHash == bytes32(0)) revert InvalidRecordHash();
        if (!activePolicies[_policyHash]) revert PolicyNotActive();
        if (audits[_auditId].exists) revert AuditAlreadyExists();

        audits[_auditId] = AuditCommitment({
            recordHash: _recordHash,
            policyHash: _policyHash,
            timestamp: block.timestamp,
            submitter: msg.sender,
            exists: true
        });

        emit AuditRecorded(
            _auditId,
            _policyHash,
            _recordHash,
            block.timestamp,
            msg.sender
        );

        return true;
    }

    /**
     * @notice Retrieves the recorded on-chain commitment for a given auditId.
     * @param _auditId Pseudonymous audit commitment identifier.
     */
    function getAudit(bytes32 _auditId)
        external
        view
        returns (
            bool exists,
            bytes32 recordHash,
            bytes32 policyHash,
            uint256 timestamp,
            address submitter
        )
    {
        AuditCommitment memory item = audits[_auditId];
        return (
            item.exists,
            item.recordHash,
            item.policyHash,
            item.timestamp,
            item.submitter
        );
    }

    /**
     * @notice Checks whether an audit commitment exists on-chain.
     * @param _auditId Pseudonymous audit commitment identifier.
     */
    function hasAudit(bytes32 _auditId) external view returns (bool) {
        return audits[_auditId].exists;
    }

    /**
     * @notice Checks whether a policy definition is active.
     * @param _policyHash SHA-256 hash of the policy specification.
     */
    function isPolicyActive(bytes32 _policyHash) external view returns (bool) {
        return activePolicies[_policyHash];
    }
}
