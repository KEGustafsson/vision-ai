// What the webapp's setup guide may say about the computer Signal K runs on.
//
// Only used to pre-select a hardware choice in the guide. It is a hint, never
// a decision: the user can pick any option. It says nothing when it cannot
// know, and it cannot know when Signal K itself runs in a container, because
// then /etc/nv_tegra_release and the CPU seen are the container's, not the
// board's, and the vision container may well run on another computer anyway.

import fs from 'fs';

export type SetupChoice = 'orin-ds' | 'xavier-ds' | 'cpu';

export interface L4tRelease {
  /** L4T major release: 36 is JetPack 6, 35 is JetPack 5. */
  major: number;
  /** e.g. "4.3" for R36.4.3; null if the file has no REVISION field. */
  revision: string | null;
}

export interface HostInfo {
  /** Node's process.arch: "arm64", "x64", ... */
  arch: string;
  l4t: L4tRelease | null;
  /** Signal K itself runs in a container: nothing below is about the board. */
  inContainer: boolean;
  /** The guide's suggested choice, or null when this host can't tell. */
  recommended: SetupChoice | null;
}

/**
 * Parse /etc/nv_tegra_release, whose first line reads e.g.
 * "# R36 (release), REVISION: 4.3, GCID: 38968081, BOARD: generic, ...".
 */
export function parseTegraRelease(text: string): L4tRelease | null {
  const m = /^#\s*R(\d+)\s*\(release\)(?:,\s*REVISION:\s*([0-9.]+))?/m.exec(text);
  if (!m) return null;
  return { major: Number(m[1]), revision: m[2] ?? null };
}

export function recommend(arch: string, l4t: L4tRelease | null, inContainer: boolean): SetupChoice | null {
  if (inContainer) return null;
  if (l4t) {
    if (l4t.major === 36) return 'orin-ds';
    if (l4t.major === 35) return 'xavier-ds';
    return null; // another JetPack: none of the published images is for it
  }
  if (arch === 'x64' || arch === 'arm64') return 'cpu';
  return null;
}

export interface HostProbe {
  arch: string;
  exists(path: string): boolean;
  read(path: string): string;
}

const realProbe: HostProbe = {
  arch: process.arch,
  exists: (p) => fs.existsSync(p),
  read: (p) => fs.readFileSync(p, 'utf8'),
};

export function readHostInfo(probe: HostProbe = realProbe): HostInfo {
  const inContainer = probe.exists('/.dockerenv') || probe.exists('/run/.containerenv');
  let l4t: L4tRelease | null = null;
  if (!inContainer) {
    try {
      if (probe.exists('/etc/nv_tegra_release')) {
        l4t = parseTegraRelease(probe.read('/etc/nv_tegra_release'));
      }
    } catch {
      l4t = null; // unreadable: say nothing rather than guess
    }
  }
  return { arch: probe.arch, l4t, inContainer, recommended: recommend(probe.arch, l4t, inContainer) };
}
