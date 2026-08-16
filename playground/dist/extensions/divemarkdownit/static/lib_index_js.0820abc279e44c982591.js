"use strict";
(self["webpackChunkdivemarkdownit"] = self["webpackChunkdivemarkdownit"] || []).push([["lib_index_js"],{

/***/ "./lib/index.js":
/*!**********************!*\
  !*** ./lib/index.js ***!
  \**********************/
/***/ ((__unused_webpack_module, __webpack_exports__, __webpack_require__) => {

__webpack_require__.r(__webpack_exports__);
/* harmony export */ __webpack_require__.d(__webpack_exports__, {
/* harmony export */   "default": () => (/* reexport safe */ _markdown_it__WEBPACK_IMPORTED_MODULE_0__["default"])
/* harmony export */ });
/* harmony import */ var _markdown_it__WEBPACK_IMPORTED_MODULE_0__ = __webpack_require__(/*! ./markdown-it */ "./lib/markdown-it/index.js");



/***/ }),

/***/ "./lib/markdown-it/index.js":
/*!**********************************!*\
  !*** ./lib/markdown-it/index.js ***!
  \**********************************/
/***/ ((__unused_webpack_module, __webpack_exports__, __webpack_require__) => {

__webpack_require__.r(__webpack_exports__);
/* harmony export */ __webpack_require__.d(__webpack_exports__, {
/* harmony export */   "default": () => (__WEBPACK_DEFAULT_EXPORT__)
/* harmony export */ });
/* harmony import */ var _jupyterlab_settingregistry__WEBPACK_IMPORTED_MODULE_0__ = __webpack_require__(/*! @jupyterlab/settingregistry */ "webpack/sharing/consume/default/@jupyterlab/settingregistry");
/* harmony import */ var _jupyterlab_settingregistry__WEBPACK_IMPORTED_MODULE_0___default = /*#__PURE__*/__webpack_require__.n(_jupyterlab_settingregistry__WEBPACK_IMPORTED_MODULE_0__);
/* harmony import */ var _jupyterlab_apputils__WEBPACK_IMPORTED_MODULE_1__ = __webpack_require__(/*! @jupyterlab/apputils */ "webpack/sharing/consume/default/@jupyterlab/apputils");
/* harmony import */ var _jupyterlab_apputils__WEBPACK_IMPORTED_MODULE_1___default = /*#__PURE__*/__webpack_require__.n(_jupyterlab_apputils__WEBPACK_IMPORTED_MODULE_1__);
/* harmony import */ var _jupyterlab_mainmenu__WEBPACK_IMPORTED_MODULE_2__ = __webpack_require__(/*! @jupyterlab/mainmenu */ "webpack/sharing/consume/default/@jupyterlab/mainmenu");
/* harmony import */ var _jupyterlab_mainmenu__WEBPACK_IMPORTED_MODULE_2___default = /*#__PURE__*/__webpack_require__.n(_jupyterlab_mainmenu__WEBPACK_IMPORTED_MODULE_2__);
/* harmony import */ var _setting__WEBPACK_IMPORTED_MODULE_6__ = __webpack_require__(/*! ./setting */ "./lib/markdown-it/setting.js");
/* harmony import */ var _manager__WEBPACK_IMPORTED_MODULE_4__ = __webpack_require__(/*! ./manager */ "./lib/markdown-it/manager.js");
/* harmony import */ var _widget__WEBPACK_IMPORTED_MODULE_5__ = __webpack_require__(/*! ./widget */ "./lib/markdown-it/widget.js");
/* harmony import */ var _values__WEBPACK_IMPORTED_MODULE_3__ = __webpack_require__(/*! ./values */ "./lib/markdown-it/values.js");
/* harmony import */ var _plugins__WEBPACK_IMPORTED_MODULE_7__ = __webpack_require__(/*! ./plugins */ "./lib/markdown-it/plugins/index.js");








const core = {
    id: `${_values__WEBPACK_IMPORTED_MODULE_3__.kPackageName}:core`,
    autoStart: true,
    provides: _values__WEBPACK_IMPORTED_MODULE_3__.IMarkdownManager,
    requires: [_jupyterlab_settingregistry__WEBPACK_IMPORTED_MODULE_0__.ISettingRegistry, _jupyterlab_apputils__WEBPACK_IMPORTED_MODULE_1__.ICommandPalette],
    optional: [_jupyterlab_mainmenu__WEBPACK_IMPORTED_MODULE_2__.IMainMenu],
    activate: (app, settings, palette, menu) => {
        const { commands, shell } = app;
        const manager = new _manager__WEBPACK_IMPORTED_MODULE_4__.MarkdownItManager();
        // set the static manager
        _widget__WEBPACK_IMPORTED_MODULE_5__.RenderedMarkdown.markdownItManager = manager;
        // eventually load settings
        settings
            .load(core.id)
            .then(settings => (manager.settings = settings))
            .catch(console.warn);
        let settingsMain;
        // commands
        commands.addCommand(_values__WEBPACK_IMPORTED_MODULE_3__.CommandID.showSettings, {
            label: 'JupyterLab MarkdownIt Settings...',
            execute: args => {
                if (settingsMain == null) {
                    const model = new _setting__WEBPACK_IMPORTED_MODULE_6__.Model();
                    model.advancedRequested.connect(() => commands.execute('settingeditor:open'));
                    model.manager = manager;
                    const content = new _setting__WEBPACK_IMPORTED_MODULE_6__.MarkdownItSettings(model);
                    settingsMain = new _jupyterlab_apputils__WEBPACK_IMPORTED_MODULE_1__.MainAreaWidget({ content });
                    settingsMain.title.label = 'JupyterLab MarkdownIt';
                    settingsMain.title.icon = _values__WEBPACK_IMPORTED_MODULE_3__.kMarkupIcon;
                    settingsMain.disposed.connect(() => (settingsMain = null));
                }
                shell.add(settingsMain, 'main');
                shell.activateById(settingsMain.id);
            }
        });
        // cached enabled setting
        let enabled = true;
        manager.settingsChanged.connect(() => {
            const { composite } = manager.settings;
            if (composite != null) {
                enabled = !!composite.enabled;
            }
        });
        commands.addCommand(_values__WEBPACK_IMPORTED_MODULE_3__.CommandID.toggleRenderer, {
            label: args => 'Use JupyterLab MarkdownIt',
            caption: 'Reopen documents to see changes',
            isToggled: () => enabled,
            isEnabled: () => manager.settings != null,
            execute: args => {
                manager.enabled = !!((args === null || args === void 0 ? void 0 : args.enabled) == null ? !enabled : args.enabled);
            }
        });
        palette.addItem({
            command: _values__WEBPACK_IMPORTED_MODULE_3__.CommandID.showSettings,
            category: _values__WEBPACK_IMPORTED_MODULE_3__.kCommandCategory
        });
        palette.addItem({
            command: _values__WEBPACK_IMPORTED_MODULE_3__.CommandID.toggleRenderer,
            category: _values__WEBPACK_IMPORTED_MODULE_3__.kCommandCategory
        });
        if (menu) {
            menu.settingsMenu.addGroup([
                { command: _values__WEBPACK_IMPORTED_MODULE_3__.CommandID.toggleRenderer },
                { command: _values__WEBPACK_IMPORTED_MODULE_3__.CommandID.showSettings }
            ], 100);
        }
        return manager;
    }
};
/* harmony default export */ const __WEBPACK_DEFAULT_EXPORT__ = ([core, ..._plugins__WEBPACK_IMPORTED_MODULE_7__.plugins]);


/***/ }),

/***/ "./lib/markdown-it/manager.js":
/*!************************************!*\
  !*** ./lib/markdown-it/manager.js ***!
  \************************************/
/***/ ((__unused_webpack_module, __webpack_exports__, __webpack_require__) => {

__webpack_require__.r(__webpack_exports__);
/* harmony export */ __webpack_require__.d(__webpack_exports__, {
/* harmony export */   "MarkdownItManager": () => (/* binding */ MarkdownItManager)
/* harmony export */ });
/* harmony import */ var _lumino_signaling__WEBPACK_IMPORTED_MODULE_0__ = __webpack_require__(/*! @lumino/signaling */ "webpack/sharing/consume/default/@lumino/signaling");
/* harmony import */ var _lumino_signaling__WEBPACK_IMPORTED_MODULE_0___default = /*#__PURE__*/__webpack_require__.n(_lumino_signaling__WEBPACK_IMPORTED_MODULE_0__);
/* harmony import */ var _jupyterlab_rendermime__WEBPACK_IMPORTED_MODULE_1__ = __webpack_require__(/*! @jupyterlab/rendermime */ "webpack/sharing/consume/default/@jupyterlab/rendermime");
/* harmony import */ var _jupyterlab_rendermime__WEBPACK_IMPORTED_MODULE_1___default = /*#__PURE__*/__webpack_require__.n(_jupyterlab_rendermime__WEBPACK_IMPORTED_MODULE_1__);
/* harmony import */ var _jupyterlab_codemirror__WEBPACK_IMPORTED_MODULE_2__ = __webpack_require__(/*! @jupyterlab/codemirror */ "webpack/sharing/consume/default/@jupyterlab/codemirror");
/* harmony import */ var _jupyterlab_codemirror__WEBPACK_IMPORTED_MODULE_2___default = /*#__PURE__*/__webpack_require__.n(_jupyterlab_codemirror__WEBPACK_IMPORTED_MODULE_2__);
/* harmony import */ var markdown_it__WEBPACK_IMPORTED_MODULE_3__ = __webpack_require__(/*! markdown-it */ "webpack/sharing/consume/default/markdown-it/markdown-it");
/* harmony import */ var markdown_it__WEBPACK_IMPORTED_MODULE_3___default = /*#__PURE__*/__webpack_require__.n(markdown_it__WEBPACK_IMPORTED_MODULE_3__);
/* harmony import */ var _widget__WEBPACK_IMPORTED_MODULE_4__ = __webpack_require__(/*! ./widget */ "./lib/markdown-it/widget.js");





const ORIGINAL_RENDERER = _jupyterlab_rendermime__WEBPACK_IMPORTED_MODULE_1__.markdownRendererFactory.createRenderer;
/**
 * An implementation of a source of markdown renderers with markdown-it and plugins
 */
class MarkdownItManager {
    constructor() {
        this.settingsChanged = new _lumino_signaling__WEBPACK_IMPORTED_MODULE_0__.Signal(this);
        /**
         * A list of plugin ids disabled by user settings
         */
        this.userDisabledPlugins = [];
        /**
         * Whether to use the markdown-it renderer: if installed, will use unless configured by user
         */
        this.useMarkdownIt = true;
        /**
         * MarkdownIt options configured by the user.
         */
        this.userMarkdownItOptions = {};
        /**
         * Per-plugin options configured by the user.
         */
        this.userPluginOptions = {};
        /**
         * Providers labeled by an arbitrary key (usually the markdown-it package name)
         */
        this._pluginProviders = new Map();
        /**
         * Create a new renderer, either with markdown-it or the original implementation
         */
        this.createRenderer = (options) => {
            return this.useMarkdownIt
                ? new _widget__WEBPACK_IMPORTED_MODULE_4__.RenderedMarkdown(options)
                : ORIGINAL_RENDERER(options);
        };
        /**
         * Use CodeMirror to highlight code blocks,
         *
         * NOTE: May be overridden by plugins
         */
        this.highlightCode = (str, lang) => {
            if (!lang) {
                return ''; // use external default escaping
            }
            try {
                const spec = _jupyterlab_codemirror__WEBPACK_IMPORTED_MODULE_2__.Mode.findBest(lang);
                if (!spec) {
                    console.warn(`No CodeMirror mode: ${lang}`);
                    return;
                }
                const el = document.createElement('div');
                try {
                    _jupyterlab_codemirror__WEBPACK_IMPORTED_MODULE_2__.Mode.run(str, spec.mime, el);
                    return el.innerHTML;
                }
                catch (err) {
                    console.warn(`Failed to highlight ${lang} code`, err);
                }
            }
            catch (err) {
                console.warn(`No CodeMirror mode: ${lang}`);
                console.warn(`Require CodeMirror mode error: ${err}`);
            }
            return '';
        };
        _jupyterlab_rendermime__WEBPACK_IMPORTED_MODULE_1__.markdownRendererFactory.createRenderer = this.createRenderer;
    }
    /**
     * Update the settings, and handle changes.
     */
    set settings(settings) {
        var _a;
        if (this._settings) {
            this._settings.changed.disconnect(this.onSettingsChanged, this);
        }
        this._settings = settings;
        if (settings != null) {
            (_a = this._settings) === null || _a === void 0 ? void 0 : _a.changed.connect(this.onSettingsChanged, this);
            this.onSettingsChanged();
        }
    }
    /**
     * The settings
     */
    get settings() {
        return this._settings;
    }
    /**
     * Update caches of settings values for new renderers
     */
    onSettingsChanged() {
        var _a, _b, _c, _d;
        const useMarkdownIt = (_a = this.settings) === null || _a === void 0 ? void 0 : _a.composite['enabled'];
        this.useMarkdownIt = useMarkdownIt == null ? true : useMarkdownIt;
        this.userMarkdownItOptions =
            ((_b = this.settings) === null || _b === void 0 ? void 0 : _b.composite['markdown-it-options']) || {};
        this.userDisabledPlugins =
            ((_c = this.settings) === null || _c === void 0 ? void 0 : _c.composite['disabled-plugins']) || [];
        this.userPluginOptions =
            ((_d = this.settings) === null || _d === void 0 ? void 0 : _d.composite['plugin-options']) || {};
        // re-brodcast settings changes
        this.settingsChanged.emit(void 0);
    }
    get enabled() {
        var _a;
        const enabled = (_a = this.settings) === null || _a === void 0 ? void 0 : _a.composite;
        return !!(enabled == null ? true : enabled);
    }
    set enabled(enabled) {
        if (this.settings == null) {
            console.warn("Can't set enabled status of markdown extensions without settings");
            return;
        }
        this.settings.set('enabled', enabled);
    }
    /**
     * Add a provider for a plugin which can be resolved lazily
     */
    addPluginProvider(provider) {
        this._pluginProviders.set(provider.id, provider);
    }
    getPluginProvider(id) {
        return this._pluginProviders.get(id);
    }
    get pluginProviderIds() {
        return [...this._pluginProviders.keys()];
    }
    /**
     * Remove a provider by name
     */
    removePluginProvider(name) {
        this._pluginProviders.delete(name);
    }
    /**
     * Get a MarkdownIt instance
     */
    async getMarkdownIt(widget, options = {}) {
        const allOptions = Object.assign(Object.assign(Object.assign({}, (await this.getOptions(widget))), options), this.userMarkdownItOptions);
        let md = new (markdown_it__WEBPACK_IMPORTED_MODULE_3___default())('default', allOptions);
        for (const [id, provider] of this._pluginProviders.entries()) {
            if (this.userDisabledPlugins.indexOf(id) !== -1) {
                continue;
            }
            try {
                const userOptions = this.userPluginOptions[id] || [];
                const [plugin, ...pluginOptions] = await provider.plugin();
                let i = 0;
                const maxOptions = Math.max(pluginOptions.length, userOptions.length);
                const compositeOptions = new Array(maxOptions);
                while (i < maxOptions) {
                    compositeOptions[i] =
                        i < userOptions.length ? userOptions[i] : pluginOptions[i];
                    i++;
                }
                md = md.use(plugin, ...compositeOptions);
            }
            catch (err) {
                console.warn(`Failed to load/use markdown-it plugin ${id}`, err);
            }
        }
        return md;
    }
    /**
     * Combine core options with base options, plugin provider options, and user settings
     */
    async getOptions(widget) {
        let allOptions = this.baseMarkdownItOptions;
        for (const [id, plugin] of this._pluginProviders.entries()) {
            if (this.userDisabledPlugins.indexOf(id) !== -1) {
                continue;
            }
            if (plugin.options == null) {
                continue;
            }
            try {
                allOptions = Object.assign(Object.assign({}, allOptions), (await plugin.options(widget)));
            }
            catch (err) {
                console.warn(`Failed to get options from markdown-it plugin ${id}`, err);
            }
        }
        return allOptions;
    }
    /**
     * Default MarkdownIt options,
     *
     * NOTE: May be overridden by plugins
     */
    get baseMarkdownItOptions() {
        return {
            html: true,
            linkify: true,
            typographer: true,
            langPrefix: `cm-s-${_jupyterlab_codemirror__WEBPACK_IMPORTED_MODULE_2__.CodeMirrorEditor.defaultConfig.theme} language-`,
            highlight: this.highlightCode
        };
    }
}


/***/ }),

/***/ "./lib/markdown-it/plugins/anchor.js":
/*!*******************************************!*\
  !*** ./lib/markdown-it/plugins/anchor.js ***!
  \*******************************************/
/***/ ((__unused_webpack_module, __webpack_exports__, __webpack_require__) => {

__webpack_require__.r(__webpack_exports__);
/* harmony export */ __webpack_require__.d(__webpack_exports__, {
/* harmony export */   "anchor": () => (/* binding */ anchor)
/* harmony export */ });
/* harmony import */ var _utils__WEBPACK_IMPORTED_MODULE_0__ = __webpack_require__(/*! ../utils */ "./lib/markdown-it/utils.js");
/* harmony import */ var _values__WEBPACK_IMPORTED_MODULE_1__ = __webpack_require__(/*! ../values */ "./lib/markdown-it/values.js");


/**
 * Adds anchors to headers
 */
const anchor = (0,_utils__WEBPACK_IMPORTED_MODULE_0__.simpleMarkdownItPlugin)(_values__WEBPACK_IMPORTED_MODULE_1__.kPackageName, {
    id: 'markdown-it-anchor',
    title: 'Heading Anchors',
    description: 'Create clickable links for header elements',
    documentationUrls: {
        Plugin: 'https://github.com/valeriangalliat/markdown-it-anchor'
    },
    plugin: async () => {
        const anchorPlugin = await __webpack_require__.e(/*! import() | markdown-it-anchor */ "markdown-it-anchor").then(__webpack_require__.t.bind(__webpack_require__, /*! markdown-it-anchor */ "webpack/sharing/consume/default/markdown-it-anchor/markdown-it-anchor", 23));
        return [
            anchorPlugin.default,
            {
                // match JupyterLab default behavior
                permalink: true,
                permalinkClass: 'jp-InternalAnchorLink',
                slugify: (title) => title.replace(/ /g, '-')
            }
        ];
    }
});


/***/ }),

/***/ "./lib/markdown-it/plugins/diagrams.js":
/*!*********************************************!*\
  !*** ./lib/markdown-it/plugins/diagrams.js ***!
  \*********************************************/
/***/ ((__unused_webpack_module, __webpack_exports__, __webpack_require__) => {

__webpack_require__.r(__webpack_exports__);
/* harmony export */ __webpack_require__.d(__webpack_exports__, {
/* harmony export */   "diagrams": () => (/* binding */ diagrams)
/* harmony export */ });
/* harmony import */ var _utils__WEBPACK_IMPORTED_MODULE_0__ = __webpack_require__(/*! ../utils */ "./lib/markdown-it/utils.js");
/* harmony import */ var _values__WEBPACK_IMPORTED_MODULE_1__ = __webpack_require__(/*! ../values */ "./lib/markdown-it/values.js");


/**
 * Draw diagrams
 */
const diagrams = (0,_utils__WEBPACK_IMPORTED_MODULE_0__.simpleMarkdownItPlugin)(_values__WEBPACK_IMPORTED_MODULE_1__.kPackageName, {
    id: 'markdown-it-diagrams',
    title: 'Drawing Diagrams',
    description: 'Draw diagrams with mermaid and railroad',
    documentationUrls: {
        Plugin: 'https://github.com/valeriangalliat/markdown-it-anchor'
    },
    examples: {
        'Railroad': `
\`\`\`railroad
Diagram(
  Optional('+', 'skip'),
  Choice(0,
    NonTerminal('name-start char'),
    NonTerminal('escape')),
  ZeroOrMore(
    Choice(0,
      NonTerminal('name char'),
      NonTerminal('escape'))))
\`\`\`
`
    },
    plugin: async () => {
        const diagramsPlugin = await __webpack_require__.e(/*! import() | markdown-it-diagrams */ "markdown-it-diagrams").then(__webpack_require__.t.bind(__webpack_require__, /*! markdown-it-diagrams */ "webpack/sharing/consume/default/markdown-it-diagrams/markdown-it-diagrams", 23));
        return [diagramsPlugin.default];
    }
});


/***/ }),

/***/ "./lib/markdown-it/plugins/docutils.js":
/*!*********************************************!*\
  !*** ./lib/markdown-it/plugins/docutils.js ***!
  \*********************************************/
/***/ ((__unused_webpack_module, __webpack_exports__, __webpack_require__) => {

__webpack_require__.r(__webpack_exports__);
/* harmony export */ __webpack_require__.d(__webpack_exports__, {
/* harmony export */   "docutils": () => (/* binding */ docutils)
/* harmony export */ });
/* harmony import */ var _utils__WEBPACK_IMPORTED_MODULE_0__ = __webpack_require__(/*! ../utils */ "./lib/markdown-it/utils.js");
/* harmony import */ var _values__WEBPACK_IMPORTED_MODULE_1__ = __webpack_require__(/*! ../values */ "./lib/markdown-it/values.js");


/**
 * Provides docutils
 */
const docutils = (0,_utils__WEBPACK_IMPORTED_MODULE_0__.simpleMarkdownItPlugin)(_values__WEBPACK_IMPORTED_MODULE_1__.kPackageName, {
    id: 'markdown-it-docutils',
    title: 'MyST',
    description: 'Use MyST syntax Markdown (limited) in JupyterLab',
    documentationUrls: {
        Plugin: 'https://github.com/executablebooks/markdown-it-docutils'
    },
    examples: {
        'Directives and roles': `
Unhandled role: {unknown}\`content\` in paragraph.

Raw role: {raw}\`content\`

Unhandled directive:

\`\`\`{unknown} argument
content
\`\`\`

Erroneous directive:

\`\`\`{admonition}
\`\`\`

Admonition with title

\`\`\`{admonition} A **Title**
Some *content*
\`\`\`

Nested admonitions:

\`\`\`\`{note} This is a note
\`\`\`{warning} This is a nested warning
\`\`\`
\`\`\`\`

Images and figures:

\`\`\`{image} https://via.placeholder.com/150
:align: center
\`\`\`

\`\`\`{figure} https://via.placeholder.com/150
:align: center

A **caption**
\`\`\`

Tables:

\`\`\`{list-table} Caption *text*
:header-rows: 1

- * Heading, Column 1
  * Heading, Column 2
- * Row, Column 1
  * Row, Column 2

    Second paragraph
\`\`\`

HTML:

* H{sub}\`2\`O
* 4{sup}\`th\` of July
* {abbr}\`CSS (Cascading Style Sheets)\`
          
Math:
          
\`\`\`{math}
:label: math_label
w_{t+1} = (1 + r_{t+1}) s(w_t) + y_{t+1}    
\`\`\`
`
    },
    plugin: async () => {
        const docutilsPlugin = await __webpack_require__.e(/*! import() | markdown-it-docutils */ "markdown-it-docutils").then(__webpack_require__.t.bind(__webpack_require__, /*! markdown-it-docutils */ "webpack/sharing/consume/default/markdown-it-docutils/markdown-it-docutils", 23));
        return [docutilsPlugin.default];
    }
});


/***/ }),

/***/ "./lib/markdown-it/plugins/footnote.js":
/*!*********************************************!*\
  !*** ./lib/markdown-it/plugins/footnote.js ***!
  \*********************************************/
/***/ ((__unused_webpack_module, __webpack_exports__, __webpack_require__) => {

__webpack_require__.r(__webpack_exports__);
/* harmony export */ __webpack_require__.d(__webpack_exports__, {
/* harmony export */   "footnote": () => (/* binding */ footnote)
/* harmony export */ });
/* harmony import */ var _utils__WEBPACK_IMPORTED_MODULE_0__ = __webpack_require__(/*! ../utils */ "./lib/markdown-it/utils.js");
/* harmony import */ var _values__WEBPACK_IMPORTED_MODULE_1__ = __webpack_require__(/*! ../values */ "./lib/markdown-it/values.js");


/**
 * Provides footnotes
 */
const footnote = (0,_utils__WEBPACK_IMPORTED_MODULE_0__.simpleMarkdownItPlugin)(_values__WEBPACK_IMPORTED_MODULE_1__.kPackageName, {
    id: 'markdown-it-footnote',
    title: 'Footnotes',
    description: 'Create links notes that appear after the current paragraph',
    documentationUrls: {
        Plugin: 'https://github.com/markdown-it/markdown-it-footnote'
    },
    examples: {
        'Footnote': `
Here is a footnote reference,[^1] and another.[^longnote]

[^1]: Here is the footnote.

[^longnote]: Here's one with multiple blocks.

  Subsequent paragraphs are indented to show that they
belong to the previous footnote.
`
    },
    plugin: async () => {
        const footnotePlugin = await __webpack_require__.e(/*! import() | markdown-it-footnote */ "markdown-it-footnote").then(__webpack_require__.t.bind(__webpack_require__, /*! markdown-it-footnote */ "webpack/sharing/consume/default/markdown-it-footnote/markdown-it-footnote", 23));
        return [footnotePlugin.default];
    }
});


/***/ }),

/***/ "./lib/markdown-it/plugins/index.js":
/*!******************************************!*\
  !*** ./lib/markdown-it/plugins/index.js ***!
  \******************************************/
/***/ ((__unused_webpack_module, __webpack_exports__, __webpack_require__) => {

__webpack_require__.r(__webpack_exports__);
/* harmony export */ __webpack_require__.d(__webpack_exports__, {
/* harmony export */   "plugins": () => (/* binding */ plugins)
/* harmony export */ });
/* harmony import */ var _anchor__WEBPACK_IMPORTED_MODULE_0__ = __webpack_require__(/*! ./anchor */ "./lib/markdown-it/plugins/anchor.js");
/* harmony import */ var _diagrams__WEBPACK_IMPORTED_MODULE_1__ = __webpack_require__(/*! ./diagrams */ "./lib/markdown-it/plugins/diagrams.js");
/* harmony import */ var _docutils__WEBPACK_IMPORTED_MODULE_2__ = __webpack_require__(/*! ./docutils */ "./lib/markdown-it/plugins/docutils.js");
/* harmony import */ var _footnote__WEBPACK_IMPORTED_MODULE_3__ = __webpack_require__(/*! ./footnote */ "./lib/markdown-it/plugins/footnote.js");
/* harmony import */ var _tasklist__WEBPACK_IMPORTED_MODULE_4__ = __webpack_require__(/*! ./tasklist */ "./lib/markdown-it/plugins/tasklist.js");





const plugins = [_anchor__WEBPACK_IMPORTED_MODULE_0__.anchor, _diagrams__WEBPACK_IMPORTED_MODULE_1__.diagrams, _docutils__WEBPACK_IMPORTED_MODULE_2__.docutils, _footnote__WEBPACK_IMPORTED_MODULE_3__.footnote, _tasklist__WEBPACK_IMPORTED_MODULE_4__.task];


/***/ }),

/***/ "./lib/markdown-it/plugins/tasklist.js":
/*!*********************************************!*\
  !*** ./lib/markdown-it/plugins/tasklist.js ***!
  \*********************************************/
/***/ ((__unused_webpack_module, __webpack_exports__, __webpack_require__) => {

__webpack_require__.r(__webpack_exports__);
/* harmony export */ __webpack_require__.d(__webpack_exports__, {
/* harmony export */   "task": () => (/* binding */ task)
/* harmony export */ });
/* harmony import */ var _utils__WEBPACK_IMPORTED_MODULE_0__ = __webpack_require__(/*! ../utils */ "./lib/markdown-it/utils.js");
/* harmony import */ var _values__WEBPACK_IMPORTED_MODULE_1__ = __webpack_require__(/*! ../values */ "./lib/markdown-it/values.js");


/**
 * Provides tasklist
 */
const task = (0,_utils__WEBPACK_IMPORTED_MODULE_0__.simpleMarkdownItPlugin)(_values__WEBPACK_IMPORTED_MODULE_1__.kPackageName, {
    id: 'markdown-it-task-checkbox',
    title: 'Tasklists',
    description: 'Create GitHub-style tasklists',
    documentationUrls: {
        Plugin: 'https://github.com/linsir/markdown-it-task-checkbox'
    },
    examples: {
        'Tasklist': `
- [ ] Mercury
- [x] Venus
- [x] Earth (Orbit/Moon)
- [x] Mars
- [ ] Jupiter
- [ ] Saturn
- [ ] Uranus
- [ ] Neptune
- [ ] Comet Haley
`
    },
    plugin: async () => {
        const taskPlugin = await __webpack_require__.e(/*! import() | markdown-it-task-checkbox */ "markdown-it-task-checkbox").then(__webpack_require__.t.bind(__webpack_require__, /*! markdown-it-task-checkbox */ "webpack/sharing/consume/default/markdown-it-task-checkbox/markdown-it-task-checkbox", 23));
        return [taskPlugin.default];
    }
});


/***/ }),

/***/ "./lib/markdown-it/setting.js":
/*!************************************!*\
  !*** ./lib/markdown-it/setting.js ***!
  \************************************/
/***/ ((__unused_webpack_module, __webpack_exports__, __webpack_require__) => {

__webpack_require__.r(__webpack_exports__);
/* harmony export */ __webpack_require__.d(__webpack_exports__, {
/* harmony export */   "MarkdownItSettings": () => (/* binding */ MarkdownItSettings),
/* harmony export */   "Model": () => (/* binding */ Model)
/* harmony export */ });
/* harmony import */ var _lumino_signaling__WEBPACK_IMPORTED_MODULE_0__ = __webpack_require__(/*! @lumino/signaling */ "webpack/sharing/consume/default/@lumino/signaling");
/* harmony import */ var _lumino_signaling__WEBPACK_IMPORTED_MODULE_0___default = /*#__PURE__*/__webpack_require__.n(_lumino_signaling__WEBPACK_IMPORTED_MODULE_0__);
/* harmony import */ var _jupyterlab_apputils__WEBPACK_IMPORTED_MODULE_1__ = __webpack_require__(/*! @jupyterlab/apputils */ "webpack/sharing/consume/default/@jupyterlab/apputils");
/* harmony import */ var _jupyterlab_apputils__WEBPACK_IMPORTED_MODULE_1___default = /*#__PURE__*/__webpack_require__.n(_jupyterlab_apputils__WEBPACK_IMPORTED_MODULE_1__);
/* harmony import */ var react__WEBPACK_IMPORTED_MODULE_2__ = __webpack_require__(/*! react */ "webpack/sharing/consume/default/react");
/* harmony import */ var react__WEBPACK_IMPORTED_MODULE_2___default = /*#__PURE__*/__webpack_require__.n(react__WEBPACK_IMPORTED_MODULE_2__);



const kClassSettings = 'jp-MarkdownItSettings';
const kClassDocs = 'jp-MarkdownItSettings-Docs';
const kClassExamples = 'jp-MarkdownItSettings-Examples';
const kClassDisabled = 'jp-mu-mod-disabled';
const kIdStem = 'id-jp-mu';
const kIdGlobal = `${kIdStem}-global`;
const kIdPlugin = `${kIdStem}-plugin`;
class Model extends _jupyterlab_apputils__WEBPACK_IMPORTED_MODULE_1__.VDomModel {
    constructor() {
        super(...arguments);
        this.advancedRequested = new _lumino_signaling__WEBPACK_IMPORTED_MODULE_0__.Signal(this);
        this._disabledPlugins = [];
        this._enabled = true;
        this._providers = [];
    }
    dispose() {
        super.dispose();
        if (this._manager) {
            this._manager.settingsChanged.disconnect(this.onSettingsChanged, this);
            this._manager = null;
        }
    }
    get manager() {
        return this._manager;
    }
    set manager(manager) {
        this._manager = manager;
        if (manager) {
            manager.settingsChanged.connect(this.onSettingsChanged, this);
            this.onSettingsChanged();
        }
    }
    get disabledPlugins() {
        return this._disabledPlugins;
    }
    get enabled() {
        return this._enabled;
    }
    set enabled(enabled) {
        this._manager.settings.set('enabled', enabled);
    }
    get providers() {
        return this._providers;
    }
    setPluginEnabled(id, enabled) {
        const disabledPlugins = this._disabledPlugins.slice();
        const idx = disabledPlugins.indexOf(id);
        if (enabled) {
            disabledPlugins.splice(idx);
        }
        else {
            disabledPlugins.push(id);
        }
        if (disabledPlugins.length) {
            this.manager.settings.set('disabled-plugins', disabledPlugins);
        }
        else {
            this.manager.settings.remove('disabled-plugins');
        }
    }
    onSettingsChanged() {
        const { composite } = this.manager.settings;
        if (composite != null) {
            this._disabledPlugins = (composite['disabled-plugins'] ||
                []);
            this._enabled = composite['enabled'];
            this._providers = this.manager.pluginProviderIds.map(this.manager.getPluginProvider, this.manager);
            this._providers.sort(this.sortByTitle);
        }
        this.stateChanged.emit(void 0);
    }
    sortByTitle(a, b) {
        return a.title.localeCompare(b.title);
    }
}
class MarkdownItSettings extends _jupyterlab_apputils__WEBPACK_IMPORTED_MODULE_1__.VDomRenderer {
    constructor(model) {
        super(model);
        this.onPluginEnabledChanged = (evt) => {
            const { value, checked } = evt.currentTarget;
            this.model.setPluginEnabled(value, checked);
        };
        this.onEnabledChanged = (evt) => {
            this.model.enabled = evt.currentTarget.checked;
        };
        this.onAdvancedClicked = () => {
            this.model.advancedRequested.emit(void 0);
        };
        this.addClass(kClassSettings);
        this.addClass('jp-RenderedHTMLCommon');
    }
    dispose() {
        var _a;
        (_a = this.model) === null || _a === void 0 ? void 0 : _a.dispose();
        super.dispose();
    }
    /**
     * Render the settings form
     */
    render() {
        const m = this.model;
        const manager = m === null || m === void 0 ? void 0 : m.manager;
        const advancedLink = (react__WEBPACK_IMPORTED_MODULE_2__.createElement("a", { href: "#", onClick: this.onAdvancedClicked }, "Open in Advanced Settings..."));
        if (manager == null) {
            return react__WEBPACK_IMPORTED_MODULE_2__.createElement("div", null);
        }
        const { providers } = m;
        return (react__WEBPACK_IMPORTED_MODULE_2__.createElement("div", null,
            react__WEBPACK_IMPORTED_MODULE_2__.createElement("header", null,
                react__WEBPACK_IMPORTED_MODULE_2__.createElement("ul", null,
                    react__WEBPACK_IMPORTED_MODULE_2__.createElement("li", null,
                        react__WEBPACK_IMPORTED_MODULE_2__.createElement("a", { href: `#${kIdGlobal}` }, "Global")),
                    react__WEBPACK_IMPORTED_MODULE_2__.createElement("li", null,
                        react__WEBPACK_IMPORTED_MODULE_2__.createElement("a", { href: `#${kIdPlugin}` }, "Markdown-it Plugins"),
                        react__WEBPACK_IMPORTED_MODULE_2__.createElement("ul", null, providers.map(this.renderPluginNav, this))),
                    react__WEBPACK_IMPORTED_MODULE_2__.createElement("li", null, advancedLink))),
            react__WEBPACK_IMPORTED_MODULE_2__.createElement("article", null,
                react__WEBPACK_IMPORTED_MODULE_2__.createElement("section", { id: kIdGlobal },
                    react__WEBPACK_IMPORTED_MODULE_2__.createElement("h3", null, "Global"),
                    react__WEBPACK_IMPORTED_MODULE_2__.createElement("label", null,
                        react__WEBPACK_IMPORTED_MODULE_2__.createElement("input", { type: "checkbox", defaultChecked: m.enabled, onChange: this.onEnabledChanged }),
                        "Use ",
                        react__WEBPACK_IMPORTED_MODULE_2__.createElement("code", null, "markdown-it")),
                    react__WEBPACK_IMPORTED_MODULE_2__.createElement("blockquote", null,
                        "Enable to use the ",
                        react__WEBPACK_IMPORTED_MODULE_2__.createElement("code", null, "markdown-it"),
                        " Markdown renderer and extensions for any new renderers.")),
                react__WEBPACK_IMPORTED_MODULE_2__.createElement("h3", { id: kIdPlugin }, "Markdown-it Plugins"),
                react__WEBPACK_IMPORTED_MODULE_2__.createElement("blockquote", null,
                    "Extensions can be individually enabled or disabled with the checkboxes below. See ",
                    advancedLink,
                    " for more fine-grained control."),
                providers.map(this.renderPluginProvider, this))));
    }
    /**
     * Render a single plugin provider nav link
     */
    renderPluginNav(provider) {
        const m = this.model;
        const enabled = m.disabledPlugins.indexOf(provider.id) === -1;
        const navClass = enabled && m.enabled ? '' : kClassDisabled;
        return (react__WEBPACK_IMPORTED_MODULE_2__.createElement("li", { key: provider.id, className: navClass },
            react__WEBPACK_IMPORTED_MODULE_2__.createElement("a", { href: `#${kIdPlugin}-${provider.id}` }, provider.title)));
    }
    /**
     * Render a single plugin provider section
     */
    renderPluginProvider(provider) {
        const m = this.model;
        const enabled = m.disabledPlugins.indexOf(provider.id) === -1;
        const sectionClass = enabled && m.enabled ? '' : kClassDisabled;
        const docs = [];
        const examples = [];
        for (const label in provider.documentationUrls) {
            docs.push(this.renderDoc(label, provider.documentationUrls[label]));
        }
        for (const label in provider.examples || {}) {
            examples.push(this.renderExample(label, provider.examples[label]));
        }
        return (react__WEBPACK_IMPORTED_MODULE_2__.createElement("section", { key: provider.id, id: `${kIdPlugin}-${provider.id}`, className: sectionClass },
            react__WEBPACK_IMPORTED_MODULE_2__.createElement("div", null,
                react__WEBPACK_IMPORTED_MODULE_2__.createElement("h4", { title: `plugin id: ${provider.id}` },
                    react__WEBPACK_IMPORTED_MODULE_2__.createElement("label", null,
                        react__WEBPACK_IMPORTED_MODULE_2__.createElement("input", { type: "checkbox", value: provider.id, defaultChecked: enabled, onChange: this.onPluginEnabledChanged }),
                        provider.title)),
                react__WEBPACK_IMPORTED_MODULE_2__.createElement("ul", { className: kClassDocs }, docs)),
            react__WEBPACK_IMPORTED_MODULE_2__.createElement("blockquote", null, provider.description),
            react__WEBPACK_IMPORTED_MODULE_2__.createElement("ul", { className: kClassExamples }, examples)));
    }
    renderDoc(label, url) {
        return (react__WEBPACK_IMPORTED_MODULE_2__.createElement("li", { key: url },
            react__WEBPACK_IMPORTED_MODULE_2__.createElement("a", { href: url, target: "_blank", rel: "noopener" }, label)));
    }
    renderExample(label, code) {
        return (react__WEBPACK_IMPORTED_MODULE_2__.createElement("div", { key: label },
            react__WEBPACK_IMPORTED_MODULE_2__.createElement("p", null,
                react__WEBPACK_IMPORTED_MODULE_2__.createElement("label", null,
                    react__WEBPACK_IMPORTED_MODULE_2__.createElement("em", null, label))),
            react__WEBPACK_IMPORTED_MODULE_2__.createElement("pre", null,
                react__WEBPACK_IMPORTED_MODULE_2__.createElement("code", null, code))));
    }
}


/***/ }),

/***/ "./lib/markdown-it/utils.js":
/*!**********************************!*\
  !*** ./lib/markdown-it/utils.js ***!
  \**********************************/
/***/ ((__unused_webpack_module, __webpack_exports__, __webpack_require__) => {

__webpack_require__.r(__webpack_exports__);
/* harmony export */ __webpack_require__.d(__webpack_exports__, {
/* harmony export */   "simpleMarkdownItPlugin": () => (/* binding */ simpleMarkdownItPlugin)
/* harmony export */ });
/* harmony import */ var _values__WEBPACK_IMPORTED_MODULE_0__ = __webpack_require__(/*! ./values */ "./lib/markdown-it/values.js");

/**
 * Convenience method for building JupyterLab MarkdownIt plugins
 */
function simpleMarkdownItPlugin(packageName, provider) {
    return {
        id: `${packageName}:${provider.id}`,
        autoStart: true,
        requires: [_values__WEBPACK_IMPORTED_MODULE_0__.IMarkdownManager],
        activate: (app, markdownIt) => {
            markdownIt.addPluginProvider(provider);
        }
    };
}


/***/ }),

/***/ "./lib/markdown-it/values.js":
/*!***********************************!*\
  !*** ./lib/markdown-it/values.js ***!
  \***********************************/
/***/ ((__unused_webpack_module, __webpack_exports__, __webpack_require__) => {

__webpack_require__.r(__webpack_exports__);
/* harmony export */ __webpack_require__.d(__webpack_exports__, {
/* harmony export */   "CommandID": () => (/* binding */ CommandID),
/* harmony export */   "IMarkdownManager": () => (/* binding */ IMarkdownManager),
/* harmony export */   "kCommandCategory": () => (/* binding */ kCommandCategory),
/* harmony export */   "kMarkupIcon": () => (/* binding */ kMarkupIcon),
/* harmony export */   "kPackageName": () => (/* binding */ kPackageName)
/* harmony export */ });
/* harmony import */ var _lumino_coreutils__WEBPACK_IMPORTED_MODULE_0__ = __webpack_require__(/*! @lumino/coreutils */ "webpack/sharing/consume/default/@lumino/coreutils");
/* harmony import */ var _lumino_coreutils__WEBPACK_IMPORTED_MODULE_0___default = /*#__PURE__*/__webpack_require__.n(_lumino_coreutils__WEBPACK_IMPORTED_MODULE_0__);
/* harmony import */ var _jupyterlab_ui_components__WEBPACK_IMPORTED_MODULE_1__ = __webpack_require__(/*! @jupyterlab/ui-components */ "webpack/sharing/consume/default/@jupyterlab/ui-components");
/* harmony import */ var _jupyterlab_ui_components__WEBPACK_IMPORTED_MODULE_1___default = /*#__PURE__*/__webpack_require__.n(_jupyterlab_ui_components__WEBPACK_IMPORTED_MODULE_1__);


const kPackageName = 'divemarkdownit';
const kCommandCategory = 'Markdown Extensions';
const kMarkupIcon = new _jupyterlab_ui_components__WEBPACK_IMPORTED_MODULE_1__.LabIcon({
    name: `${kPackageName}:core`,
    svgstr: _jupyterlab_ui_components__WEBPACK_IMPORTED_MODULE_1__.markdownIcon.svgstr.replace('jp-icon-contrast0', 'jp-icon-contrast1')
});
const IMarkdownManager = new _lumino_coreutils__WEBPACK_IMPORTED_MODULE_0__.Token(kPackageName);
var CommandID;
(function (CommandID) {
    CommandID.showSettings = 'markdown-it:show-settings';
    CommandID.toggleRenderer = 'markdown-it:toggle-renderer';
})(CommandID || (CommandID = {}));


/***/ }),

/***/ "./lib/markdown-it/widget.js":
/*!***********************************!*\
  !*** ./lib/markdown-it/widget.js ***!
  \***********************************/
/***/ ((__unused_webpack_module, __webpack_exports__, __webpack_require__) => {

__webpack_require__.r(__webpack_exports__);
/* harmony export */ __webpack_require__.d(__webpack_exports__, {
/* harmony export */   "RenderedMarkdown": () => (/* binding */ RenderedMarkdown)
/* harmony export */ });
/* harmony import */ var _jupyterlab_rendermime__WEBPACK_IMPORTED_MODULE_0__ = __webpack_require__(/*! @jupyterlab/rendermime */ "webpack/sharing/consume/default/@jupyterlab/rendermime");
/* harmony import */ var _jupyterlab_rendermime__WEBPACK_IMPORTED_MODULE_0___default = /*#__PURE__*/__webpack_require__.n(_jupyterlab_rendermime__WEBPACK_IMPORTED_MODULE_0__);


/**
 * A mime renderer for displaying Markdown with embedded latex.
 */
class RenderedMarkdown extends _jupyterlab_rendermime__WEBPACK_IMPORTED_MODULE_0__.RenderedHTMLCommon {
    constructor(options) {
        super(options);
        this.addClass('jp-RenderedMarkdown');
    }
    /**
     * Render a mime model.
     *
     * @param model - The mime model to render.
     *
     * @returns A promise which resolves when rendering is complete.
     */
    async render(model) {
        if (this.md == null) {
            this.md = await RenderedMarkdown.markdownItManager.getMarkdownIt(this);
        }
        let markdown = String(model.data[this.mimeType]);
        let text_and_math = (0,_jupyterlab_rendermime__WEBPACK_IMPORTED_MODULE_0__.removeMath)(markdown);
        let text = text_and_math['text'];
        let html = this.md.render(text);
        html = (0,_jupyterlab_rendermime__WEBPACK_IMPORTED_MODULE_0__.replaceMath)(html, text_and_math['math']);
        return await (0,_jupyterlab_rendermime__WEBPACK_IMPORTED_MODULE_0__.renderHTML)({
            host: this.node,
            source: html,
            trusted: model.trusted,
            resolver: this.resolver,
            sanitizer: this.sanitizer,
            linkHandler: this.linkHandler,
            shouldTypeset: this.isAttached,
            latexTypesetter: this.latexTypesetter
        });
    }
    /**
     * A message handler invoked on an `'after-attach'` message.
     */
    onAfterAttach(msg) {
        if (this.latexTypesetter) {
            this.latexTypesetter.typeset(this.node);
        }
    }
}


/***/ })

}]);
//# sourceMappingURL=lib_index_js.0820abc279e44c982591.js.map