/**
 * API Key Encryption Utility
 * 
 * Encrypts/decrypts exchange API keys before storing in PostgreSQL.
 * Uses AES-256-GCM for authenticated encryption.
 * 
 * The ENCRYPTION_KEY env var must be a 32-byte hex string (64 chars).
 * Generate one with: node -e "console.log(require('crypto').randomBytes(32).toString('hex'))"
 */
import crypto from 'crypto';

const ALGORITHM = 'aes-256-gcm';
const IV_LENGTH = 16;
const AUTH_TAG_LENGTH = 16;

function getKey(): Buffer {
    const key = process.env.ENCRYPTION_KEY;
    if (!key || key.length !== 64) {
        throw new Error(
            'ENCRYPTION_KEY must be set as a 64-char hex string (32 bytes). ' +
            'Generate with: node -e "console.log(require(\'crypto\').randomBytes(32).toString(\'hex\'))"'
        );
    }
    return Buffer.from(key, 'hex');
}

export function encrypt(plaintext: string): { encrypted: string; iv: string } {
    const key = getKey();
    const iv = crypto.randomBytes(IV_LENGTH);

    const cipher = crypto.createCipheriv(ALGORITHM, key, iv);
    let encrypted = cipher.update(plaintext, 'utf8', 'hex');
    encrypted += cipher.final('hex');

    const authTag = cipher.getAuthTag();

    return {
        encrypted: encrypted + ':' + authTag.toString('hex'),
        iv: iv.toString('hex'),
    };
}

export function decrypt(encryptedData: string, ivHex: string): string {
    const key = getKey();
    const iv = Buffer.from(ivHex, 'hex');

    const [encrypted, authTagHex] = encryptedData.split(':');
    const authTag = Buffer.from(authTagHex, 'hex');

    const decipher = crypto.createDecipheriv(ALGORITHM, key, iv);
    decipher.setAuthTag(authTag);

    let decrypted = decipher.update(encrypted, 'hex', 'utf8');
    decrypted += decipher.final('utf8');

    return decrypted;
}

/**
 * Encrypt an API key pair for storage
 */
export function encryptApiKeys(apiKey: string, apiSecret: string) {
    const encKey = encrypt(apiKey);
    const encSecret = encrypt(apiSecret);

    return {
        apiKey: encKey.encrypted,
        apiSecret: encSecret.encrypted,
        // Store both IVs together (comma-separated)
        encryptionIv: `${encKey.iv},${encSecret.iv}`,
    };
}

/**
 * Decrypt an API key pair from storage
 */
export function decryptApiKeys(
    encApiKey: string,
    encApiSecret: string,
    encryptionIv: string
): { apiKey: string; apiSecret: string } {
    const [keyIv, secretIv] = encryptionIv.split(',');

    return {
        apiKey: decrypt(encApiKey, keyIv),
        apiSecret: decrypt(encApiSecret, secretIv),
    };
}
