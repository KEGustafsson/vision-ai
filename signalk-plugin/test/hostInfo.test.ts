import { describe, expect, it } from 'vitest';
import { HostProbe, parseTegraRelease, readHostInfo, recommend } from '../src/hostInfo';

const ORIN = '# R36 (release), REVISION: 4.3, GCID: 38968081, BOARD: generic, EABI: aarch64, DATE: Wed Jan  8 01:49:37 UTC 2025\n';
const XAVIER = '# R35 (release), REVISION: 6.4, GCID: 12345, BOARD: t186ref, EABI: aarch64, DATE: Tue\n';

function probe(files: Record<string, string>, arch = 'arm64'): HostProbe {
  return { arch, exists: (p) => p in files, read: (p) => files[p] };
}

describe('parseTegraRelease', () => {
  it('reads the L4T major release and revision', () => {
    expect(parseTegraRelease(ORIN)).toEqual({ major: 36, revision: '4.3' });
    expect(parseTegraRelease(XAVIER)).toEqual({ major: 35, revision: '6.4' });
  });
  it('accepts a file without a revision and rejects anything else', () => {
    expect(parseTegraRelease('# R36 (release)\n')).toEqual({ major: 36, revision: null });
    expect(parseTegraRelease('')).toBeNull();
    expect(parseTegraRelease('Ubuntu 22.04')).toBeNull();
  });
});

describe('recommend', () => {
  it('maps JetPack 6 to Orin DeepStream and JetPack 5 to Xavier DeepStream', () => {
    expect(recommend('arm64', { major: 36, revision: '4.3' }, false)).toBe('orin-ds');
    expect(recommend('arm64', { major: 35, revision: '6.4' }, false)).toBe('xavier-ds');
  });
  it('says nothing for a JetPack no published image is for', () => {
    expect(recommend('arm64', { major: 32, revision: '7.1' }, false)).toBeNull();
  });
  it('suggests CPU on a computer that is not a Jetson', () => {
    expect(recommend('x64', null, false)).toBe('cpu');
    expect(recommend('arm64', null, false)).toBe('cpu');
    expect(recommend('ia32', null, false)).toBeNull();
  });
  it('says nothing when Signal K runs in a container', () => {
    expect(recommend('arm64', { major: 36, revision: '4.3' }, true)).toBeNull();
    expect(recommend('x64', null, true)).toBeNull();
  });
});

describe('readHostInfo', () => {
  it('reads the board on a Jetson', () => {
    expect(readHostInfo(probe({ '/etc/nv_tegra_release': ORIN }))).toEqual({
      arch: 'arm64', l4t: { major: 36, revision: '4.3' }, inContainer: false, recommended: 'orin-ds',
    });
  });
  it("ignores the release file inside a container: it is the container's, not the board's", () => {
    const info = readHostInfo(probe({ '/.dockerenv': '', '/etc/nv_tegra_release': ORIN }));
    expect(info).toEqual({ arch: 'arm64', l4t: null, inContainer: true, recommended: null });
  });
  it('treats an unreadable release file as unknown', () => {
    const p: HostProbe = { arch: 'arm64', exists: () => true, read: () => { throw new Error('EACCES'); } };
    // exists() is true for every path here, including /.dockerenv
    expect(readHostInfo({ ...p, exists: (path) => path === '/etc/nv_tegra_release' }).l4t).toBeNull();
  });
});
