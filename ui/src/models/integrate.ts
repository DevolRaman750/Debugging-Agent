// ══════════════════════════════════════════════════════════════
// Integration Token Models
// ══════════════════════════════════════════════════════════════

export enum ResourceType {
  GITHUB = "github",
  NOTION = "notion",
  SLACK = "slack",
  OPENAI = "openai",
  ROOTIX = "rootix",
}

export interface TokenResource {
  token?: string | null;
  resourceType: ResourceType;
}
