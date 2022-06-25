<template>
    <svg class="selection-svg" :id="'selection-svg-'+id" width="95%" height="40" ref="svg" @click="selectSvg">
    <text x="20" y="0" dy="22.5" text-anchor="start" font-size="15" font-weight="normal" font-family="Comic Sans MS">Color Set {{id}}:</text>
    <text v-if="showNodes.length === 0" x="120" y="0" dy="22.5"
        text-anchor="start" font-size="15" font-weight="normal" font-family="Comic Sans MS">Empty</text>
        <g id="main-g">
            <g id="selection-g" transform="translate(10, 30)"></g>
        </g>
    </svg>
</template>

<script>
import {mapGetters, mapMutations} from 'vuex';
import * as d3 from 'd3';
window.d3 = d3;
import Util from './Util.vue';
import GlobalVar from './GlovalVar.vue';
import clone from 'just-clone';

export default {
    name: 'Selection',
    mixins: [Util, GlobalVar],
    props: {
        confusionMatrix: {
            type: Array,
            default: undefined,
        },
        id: {
            type: Number,
            default: 0,
        },
    },
    computed: {
        ...mapGetters([
            'labelHierarchy',
            'labelnames',
        ]),
        rawHierarchy: function() {
            return this.labelHierarchy;
        },
        svg: function() {
            return d3.select('#selection-svg-'+this.id);
        },
        mainG: function() {
            return this.svg.select('#main-g');
        },
        horizonTextG: function() {
            return this.svg.select('#selection-g');
        },
        colorCellSize: function() {
            return this.linesize*0.7;
        },
        svgWidth: function() {
            return this.leftCornerSize;
        },
        svgHeight: function() {
            return this.showNodes.length * this.linesize+40;
        },
        maxHorizonTextWidth: function() {
            let maxwidth = 0;
            for (const node of this.showNodes) {
                const textwidth = this.getTextWidth(node.name,
                    `${this.horizonTextAttrs['font-weight']} ${this.horizonTextAttrs['font-size']}px ${this.horizonTextAttrs['font-family']}`);
                const arrowIconNum = node.children.length===0?node.depth:node.depth+1;
                maxwidth = Math.max(maxwidth, this.horizonTextAttrs['leftMargin']*node.depth + textwidth +
                    arrowIconNum*(this.horizonTextAttrs['font-size'] + this.horizonTextAttrs['iconMargin'])+this.colorCellSize+this.colorCellMargin);
            }
            return maxwidth;
        },
        leftCornerSize: function() {
            return this.maxHorizonTextWidth;
        },
    },
    watch: {
        labelHierarchy: function(newLabelHierarchy, oldLabelHierarchy) {
            this.hierarchy = this.getHierarchy(newLabelHierarchy);
            for (const child of this.hierarchy) {
                this.initColor(child, this.defaultColor);
            }
            this.setHierarchyColors(clone(this.hierarchyColors));
            this.getDataAndRender();
        },
        confusionMatrix: function() {
            this.getDataAndRender();
        },
    },
    data: function() {
        return {
            hierarchy: {},
            showNodes: [],
            horizonTextinG: undefined,
            horizonTextAttrs: {
                'gClass': 'horizon-one-line-g',
                'leftMargin': 30,
                'text-anchor': 'start',
                'font-family': 'Comic Sans MS',
                'font-weight': 'normal',
                'font-size': 15,
                'iconMargin': 5,
                'iconDy': 3,
                'indent-line-stroke': 'gray',
                'indent-line-stroke-width': 2,
            },
            cellAttrs: {
                'gClass': 'one-cell-g',
                'size': 30,
                'stroke-width': '1px',
                'stroke': 'gray',
                'slash-text-stroke': 'gray',
                'text-fill': '#FF6A6A',
                'text-anchor': 'start',
                'font-family': 'Comic Sans MS',
                'font-weight': 'normal',
                'font-size': 15,
                'cursor': 'pointer',
                'direction-color': 'currentColor',
            },
            linesize: 30,
            colorCellMargin: 10,
            editColorSize: 15,
            textMatrixMargin: 10,
            baseColors: ['#1b9e77', '#d95f02', '#7570b3', '#e7298a', '#66a61e', '#e6ab02', '#a6761d'],
            defaultColor: 'gray',
            nextColor: 0,
            coloredNodes: [],
            hierarchyColors: {},
        };
    },
    methods: {
        ...mapMutations([
            'setHierarchyColors',
        ]),
        findNodeByName: function(name) {
            const dfs = function(root) {
                if (root.name===name) {
                    return root;
                }
                for (const node of root.children) {
                    const res = dfs(node);
                    if (res !== null) {
                        return res;
                    }
                }
                return null;
            };
            for (const root of this.hierarchy) {
                const res = dfs(root);
                if (res !== null) {
                    return res;
                }
            }
            return null;
        },
        initColor: function(root, color) {
            if (typeof(root)==='string') {
                this.hierarchyColors[root] = color;
                return color;
            }
            this.hierarchyColors[root.name] = color;
            for (const child of root.children) {
                this.initColor(child, color);
            }
        },
        setNewColor: function(root, replace=true) {
            const nodename = root.name;
            const that = this;
            const dfs = function(root, nodename) {
                if (root.name === nodename) {
                    that.initColor(root, that.baseColors[that.nextColor]);
                    that.nextColor++;
                    if (that.nextColor===that.baseColors.length) {
                        that.nextColor = 0;
                    }
                    return true;
                }
                for (const child of root.children) {
                    if (dfs(child, nodename)) {
                        that.hierarchyColors[root.name] = that.hierarchyColors[child.name];
                        return true;
                    }
                }
                return false;
            };
            if (!replace && that.hierarchyColors[root.name] !== that.defaultColor) {
                return;
            }
            for (const root of this.hierarchy) {
                dfs(root, nodename);
            }
            that.setHierarchyColors(clone(that.hierarchyColors));
            that.nextColor++;
            if (that.nextColor===that.baseColors.length) {
                that.nextColor = 0;
            }
            that.getDataAndRender();
        },
        getColoredNodes: function() {
            this.coloredNodes = [];
            const that = this;
            const dfs = function(root) {
                if (typeof(root)==='string') {
                    if (that.hierarchyColors[root] != that.defaultColor) {
                        that.coloredNodes.push(root);
                    }
                    return;
                }
                if (that.hierarchyColors[root.name] != that.defaultColor) {
                    that.coloredNodes.push(root);
                    return;
                } else {
                    for (const node of root.children) {
                        dfs(node);
                    }
                }
            };

            for (const root of this.hierarchy) {
                dfs(root);
            }
            return this.coloredNodes;
        },
        getHierarchy: function(hierarchy) {
            hierarchy = clone(hierarchy);
            const postorder = function(root, depth) {
                if (typeof(root) !== 'object') {
                    return {
                        name: root,
                        expand: false,
                        leafs: [root],
                        children: [],
                        depth: depth,
                    };
                }
                root.expand = false;
                root.depth = depth;
                let leafs = [];
                const newChildren = [];
                for (const child of root.children) {
                    const newChild = postorder(child, depth+1);
                    leafs = leafs.concat(newChild.leafs);
                    newChildren.push(newChild);
                }
                root.children = newChildren;
                root.leafs = leafs;
                return root;
            };
            for (let i=0; i<hierarchy.length; i++) {
                hierarchy[i] = postorder(hierarchy[i], 0);
            }
            return hierarchy;
        },
        getShowNodes: function(hierarchy) {
            const showNodes = [];
            const stack = Object.values(hierarchy).reverse();
            while (stack.length>0) {
                const top = stack.pop();
                showNodes.push(top);
                if (top.expand) {
                    for (let i=top.children.length-1; i>=0; i--) {
                        stack.push(top.children[i]);
                    }
                }
            }
            return showNodes;
        },
        getDataAndRender: function() {
            if (this.confusionMatrix===undefined || this.labelHierarchy===undefined) {
                return;
            }
            // get nodes to show
            this.showNodes = this.getShowNodes(this.getColoredNodes());
            this.render();
        },
        render: async function() {
            this.horizonTextinG = this.horizonTextG.selectAll('g.'+this.horizonTextAttrs['gClass']).data(this.showNodes, (d)=>d.name);

            await this.remove();
            this.transform();
            await this.update();
            await this.create();
        },
        create: async function() {
            const that = this;
            return new Promise((resolve, reject) => {
                const horizonTextinG = that.horizonTextinG.enter()
                    .append('g')
                    .attr('class', that.horizonTextAttrs['gClass'])
                    .attr('opacity', 0)
                    .attr('transform', (d, i) => `translate(${d.depth*that.horizonTextAttrs['leftMargin']}, 
                        ${i*that.linesize})`);

                horizonTextinG.transition()
                    .duration(that.createDuration)
                    .attr('opacity', 1)
                    .on('end', resolve);

                horizonTextinG.append('text')
                    .attr('x', (d) => (d.children.length===0?0:that.horizonTextAttrs['font-size'] +
                        that.horizonTextAttrs['iconMargin'])+that.colorCellSize + that.colorCellMargin*2+that.editColorSize)
                    .attr('y', 0)
                    .attr('dy', that.linesize/2+that.horizonTextAttrs['font-size']/2)
                    .attr('text-anchor', that.horizonTextAttrs['text-anchor'])
                    .attr('font-size', that.horizonTextAttrs['font-size'])
                    .attr('font-weight', that.horizonTextAttrs['font-weight'])
                    .attr('font-family', that.horizonTextAttrs['font-family'])
                    .text((d) => d.name);

                const icony = that.linesize/2-that.horizonTextAttrs['font-size']/2+that.horizonTextAttrs['iconDy'];
                horizonTextinG.filter((d) => d.children.length>0)
                    .append('image')
                    .attr('class', 'zoom')
                    .attr('xlink:href', (d) => '/static/images/'+(d.children.length>1?'arrow.png':'dot.png'))
                    .attr('x', 0)
                    .attr('y', icony)
                    .attr('width', that.horizonTextAttrs['font-size'])
                    .attr('height', that.horizonTextAttrs['font-size'])
                    .attr('transform', (d) => `rotate(${d.expand?90:0} 
                        ${that.horizonTextAttrs['font-size']/2} ${icony+that.horizonTextAttrs['font-size']/2})`)
                    .attr('cursor', 'pointer')
                    .on('click', function(e, d) {
                        if (d.children.length===1) return;
                        d.expand = !d.expand;
                        that.getDataAndRender();
                    });

                horizonTextinG
                    .append('image')
                    .attr('xlink:href', (d) => '/static/images/wrench.svg')
                    .attr('x', (d) => d.children.length===0?0:that.horizonTextAttrs['font-size'] +
                        that.horizonTextAttrs['iconMargin'])
                    .attr('y', (that.linesize-that.editColorSize)/2+that.horizonTextAttrs['iconDy'])
                    .attr('width', that.editColorSize)
                    .attr('height', that.editColorSize)
                    .attr('cursor', 'pointer')
                    .on('click', function(e, d) {
                        that.setNewColor(d);
                    });

                horizonTextinG.append('rect')
                    .attr('x', (d) => (d.children.length===0?0:that.horizonTextAttrs['font-size'] +
                        that.horizonTextAttrs['iconMargin'])+that.editColorSize+that.colorCellMargin)
                    .attr('y', (that.linesize-that.colorCellSize)/2+that.horizonTextAttrs['iconDy'])
                    .attr('width', that.colorCellSize)
                    .attr('height', that.colorCellSize)
                    .attr('stroke', that.cellAttrs['stroke'])
                    .attr('stroke-width', that.cellAttrs['stroke-width'])
                    .attr('fill', (d) => that.hierarchyColors[d.name]);

                horizonTextinG.filter((d) => d.children.length>0)
                    .append('path')
                    .attr('class', 'child-path')
                    .attr('stroke', that.horizonTextAttrs['indent-line-stroke'])
                    .attr('stroke-width', that.horizonTextAttrs['indent-line-stroke-width'])
                    .attr('d', (d)=>{
                        // find expand child length
                        const stack = [d];
                        let expandlen = 0;
                        if (d.expand) {
                            while (stack.length > 0) {
                                const top = stack.pop();
                                for (const child of top.children) {
                                    if (child.children.length>0 && child.expand===true) {
                                        stack.push(child);
                                    }
                                    expandlen++;
                                }
                            }
                        }
                        const linelen = that.linesize*expandlen;
                        const x = that.horizonTextAttrs['font-size']/2;
                        return `M ${x} ${that.linesize} L ${x} ${that.linesize+linelen}`;
                    });

                if (that.horizonTextinG.enter().size() === 0) {
                    resolve();
                }
            });
        },
        update: async function() {
            const that = this;
            return new Promise((resolve, reject) => {
                that.horizonTextinG
                    .transition()
                    .duration(that.updateDuration)
                    .attr('transform', (d, i) => `translate(${d.depth*that.horizonTextAttrs['leftMargin']}, 
                        ${i*that.linesize})`)
                    .on('end', resolve);

                const icony = that.linesize/2-that.horizonTextAttrs['font-size']/2+that.horizonTextAttrs['iconDy'];
                that.horizonTextinG.filter((d) => d.children.length>0)
                    .selectAll('image.zoom')
                    .attr('transform', (d) => `rotate(${d.expand?90:0} 
                        ${that.horizonTextAttrs['font-size']/2} ${icony+that.horizonTextAttrs['font-size']/2})`);

                that.horizonTextinG.select('rect')
                    .attr('fill', (d) => that.hierarchyColors[d.name]);

                that.horizonTextinG.filter((d) => d.children.length>0)
                    .selectAll('path.child-path')
                    .attr('stroke', that.horizonTextAttrs['indent-line-stroke'])
                    .attr('stroke-width', that.horizonTextAttrs['indent-line-stroke-width'])
                    .transition()
                    .duration(that.updateDuration)
                    .attr('d', (d)=>{
                        // find expand child length
                        const stack = [d];
                        let expandlen = 0;
                        if (d.expand) {
                            while (stack.length > 0) {
                                const top = stack.pop();
                                for (const child of top.children) {
                                    if (child.children.length>0 && child.expand===true) {
                                        stack.push(child);
                                    }
                                    expandlen++;
                                }
                            }
                        }
                        const linelen = that.linesize*expandlen;
                        const x = that.horizonTextAttrs['font-size']/2;
                        return `M ${x} ${that.linesize} L ${x} ${that.linesize+linelen}`;
                    })
                    .on('end', resolve);

                if (that.horizonTextinG.size() === 0) {
                    resolve();
                }
            });
        },
        remove: async function() {
            const that = this;
            return new Promise((resolve, reject) => {
                that.horizonTextinG.exit()
                    .transition()
                    .duration(that.removeDuration)
                    .attr('opacity', 0)
                    .remove()
                    .on('end', resolve);
                if (that.horizonTextinG.exit().size() === 0) {
                    resolve();
                }
            });
        },
        transform: async function() {
            const that = this;
            return new Promise((resolve, reject) => {
                // compute transform
                const svgRealWidth = that.$refs.svg.clientWidth;
                const svgRealHeight = that.$refs.svg.clientHeight;
                console.log(svgRealWidth, svgRealHeight);
                console.log(that.svgWidth);
                that.svg.transition()
                    .duration(that.transformDuration)
                    .attr('height', that.svgHeight)
                    .on('end', resolve);
                that.mainG.transition()
                    .duration(that.transformDuration)
                    .attr('transform', `translate(${10} ${30}) scale(1)`)
                    .on('end', resolve);
                that.horizonTextG.transition()
                    .duration(that.transformDuration)
                    .attr('transform', `translate(0, 0)`)
                    .on('end', resolve);
            });
        },
        selectHandle: function() {
            this.svg.style('border-color', 'cornflowerblue');
        },
        unselectHandle: function() {
            this.svg.style('border-color', '#c1c1c1');
        },
        selectSvg: function() {
            console.log('select svg', this.id);
            this.$emit('selectSvg', this.id);
        },
    },
    mounted: function() {
        if (this.labelHierarchy === undefined) {
            return;
        }
        this.hierarchy = this.getHierarchy(this.labelHierarchy);
        for (const child of this.hierarchy) {
            this.initColor(child, this.defaultColor);
        }
        this.setHierarchyColors(clone(this.hierarchyColors));
        this.getDataAndRender();
    },
};
</script>

<style scoped>
.selection-svg {
    border: 1px solid #c1c1c1;
    border-radius: 5px;
    margin: 5px 0 5px 0;
}
</style>
