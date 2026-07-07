// Runtime configuration for the VIQ AI frontend (loaded before the app script).
//
// EMBEDDED_MODE hides the header, footer, vessel type filter, and the
// "Additional Context" panel in search results so the app can be embedded
// (e.g. in an iframe) on another website. Default is false (full standalone UI).
// All backend/search/AI functionality is unaffected either way.
//
// To turn this on for a container built for website embedding, overwrite or
// volume-mount this file with `window.EMBEDDED_MODE = true;`, or have your
// container's entrypoint generate it from the EMBEDDED_MODE environment
// variable (e.g. with envsubst), while leaving the production container's
// copy of this file untouched.
window.EMBEDDED_MODE = true;
