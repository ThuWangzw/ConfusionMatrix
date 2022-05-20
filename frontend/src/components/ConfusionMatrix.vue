<template>
    <svg id="confusion-svg" width="100%" height="100%" ref="svg">
        <defs>
            <!-- arrowhead marker definition -->
            <marker id="arrow" viewBox="0 0 4 4" refX="2" refY="2"
                markerWidth="3" markerHeight="3"
                orient="auto-start-reverse">
            <path d="M 0 0 L 4 2 L 0 4 z" />
            </marker>
        </defs>
        <g id="main-g" transform="translate(0,0)">
            <g id="legend-g" transform="translate(0,0)"></g>
            <g id="horizon-text-g" transform="translate(0, 0)">
                <text id="horizon-legend" transform="translate(0,0) rotate(270)" text-anchor="middle" font-size="15" opacity="0"
                    font-family="Comic Sans MS" font-weight="normal">Ground Truth</text>
            </g>
            <g id="vertical-text-g" transform="translate(0, 0) rotate(-90)">
                <text id="vertical-legend" transform="translate(0,0) rotate(90)" opacity="0"
                    text-anchor="middle" font-size="15" font-family="Comic Sans MS" font-weight="normal">Prediction</text>
            </g>
            <g id="matrix-cells-g" transform="translate(0, 0)"></g>
        </g>
    </svg>
</template>

<script>
import {mapGetters} from 'vuex';
import * as d3 from 'd3';
window.d3 = d3;
import Util from './Util.vue';
import GlobalVar from './GlovalVar.vue';
import clone from 'just-clone';

export default {
    name: 'ConfusionMatrix',
    mixins: [Util, GlobalVar],
    props: {
        confusionMatrix: {
            type: Array,
            default: undefined,
        },
        returnMode: {
            type: String,
            default: 'count',
        },
        showDirection: {
            type: Boolean,
            default: false,
        },
    },
    computed: {
        ...mapGetters([
            'labelHierarchy',
            'labelnames',
        ]),
        baseMatrix: function() {
            return this.confusionMatrix;
        },
        indexNames: function() {
            return this.labelnames;
        },
        rawHierarchy: function() {
            return this.labelHierarchy;
        },
        name2index: function() {
            const result = {};
            for (let i=0; i<this.indexNames.length; i++) {
                result[this.indexNames[i]] = i;
            }
            return result;
        },
        svg: function() {
            return d3.select('#confusion-svg');
        },
        matrixWidth: function() {
            return this.showNodes.length * this.cellAttrs['size'];
        },
        svgWidth: function() {
            return this.leftCornerSize+this.textMatrixMargin+this.matrixWidth+50;// legend height
        },
        colorCellSize: function() {
            return this.showColor?this.cellAttrs['size']*0.7:0;
        },
        colorCellMargin: function() {
            return this.showColor?10:0;
        },
        horizonTextG: function() {
            return d3.select('g#horizon-text-g');
        },
        verticalTextG: function() {
            return d3.select('g#vertical-text-g');
        },
        matrixCellsG: function() {
            return d3.select('g#matrix-cells-g');
        },
        legendG: function() {
            return d3.select('g#legend-g');
        },
        mainG: function() {
            return d3.select('g#main-g');
        },
        horizonLegend: function() {
            return d3.select('text#horizon-legend');
        },
        verticalLegend: function() {
            return d3.select('text#vertical-legend');
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
        legendWidth: function() {
            return Math.min(250, this.matrixWidth);
        },
        leftCornerSize: function() {
            return this.maxHorizonTextWidth;
        },
        colorScale: function() {
            if (this.returnMode==='count') {
                return d3.scaleSequential([0, this.maxCellValue], ['rgb(255, 255, 255)', 'rgb(8, 48, 107)']).clamp(true);
            } else if (this.returnMode==='avg_iou' || this.returnMode==='avg_acc') {
                return d3.scaleSequential([1, 0], ['rgb(255, 255, 255)', 'rgb(8, 48, 107)']).clamp(true);
            } else {
                return d3.scaleSequential([0, 1], ['rgb(255, 255, 255)', 'rgb(8, 48, 107)']).clamp(true);
            }
        },
    },
    mounted: function() {
        // init legend
        this.horizonLegend.attr('opacity', 0);
        this.verticalLegend.attr('opacity', 0);
        this.hierarchy = this.getHierarchy(this.rawHierarchy);
        this.getDataAndRender();
    },
    watch: {
        labelHierarchy: function(newLabelHierarchy, oldLabelHierarchy) {
            this.hierarchy = this.getHierarchy(newLabelHierarchy);
            this.getDataAndRender();
        },
        confusionMatrix: function() {
            this.getDataAndRender();
        },
        showDirection: function() {
            this.getDataAndRender();
        },
    },
    data: function() {
        return {
            hierarchy: {},
            // layout
            textGWidth: 0,
            cellWidth: 10,
            textMatrixMargin: 10,
            showNodes: [],
            cells: [],
            // layout elements
            horizonTextinG: null,
            verticalTextinG: null,
            matrixCellsinG: null,
            // render attrs
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
            verticalTextAttrs: {
                'gClass': 'vertical-one-line-g',
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
            legendExist: false,
            // buffer
            maxCellValue: 0,
            maxCellDirection: 0,
        };
    },
    methods: {
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
            this.showNodes = this.getShowNodes(this.hierarchy);
            // get cells to render
            this.cells = [];
            this.maxCellValue = 0;
            this.maxCellDirection = 0;
            for (let i=0; i<this.showNodes.length; i++) {
                const nodea = this.showNodes[i];
                for (let j=0; j<this.showNodes.length; j++) {
                    const nodeb = this.showNodes[j];
                    const cell = {
                        key: nodea.name+','+nodeb.name,
                        info: this.getTwoCellConfusion(nodea, nodeb),
                        row: i,
                        column: j,
                        rowNode: nodea,
                        colNode: nodeb,
                    };
                    this.cells.push(cell);
                    if (!this.isHideCell(cell) && (i!=j || this.returnMode!=='count') && i!==this.showNodes.length-1 && j!==this.showNodes.length-1) {
                        this.maxCellValue = Math.max(this.maxCellValue, cell.info.val);
                        if (cell.info.direction!==undefined) this.maxCellDirection = Math.max(this.maxCellDirection, d3.sum(cell.info.direction));
                    }
                }
            }
            this.render();
        },
        render: async function() {
            this.horizonTextinG = this.horizonTextG.selectAll('g.'+this.horizonTextAttrs['gClass']).data(this.showNodes, (d)=>d.name);
            this.verticalTextinG = this.verticalTextG.selectAll('g.'+this.verticalTextAttrs['gClass']).data(this.showNodes, (d)=>d.name);
            this.matrixCellsinG = this.matrixCellsG.selectAll('g.'+this.cellAttrs['gClass']).data(this.cells, (d)=>d.key);
            if (!this.legendExist) {
                this.drawLegend();
                this.legendExist = true;
            }

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
                        ${i*that.cellAttrs['size']})`);

                horizonTextinG.transition()
                    .duration(that.createDuration)
                    .attr('opacity', 1)
                    .on('end', resolve);

                horizonTextinG.append('text')
                    .attr('x', (d) => (d.children.length===0?0:that.horizonTextAttrs['font-size'] +
                        that.horizonTextAttrs['iconMargin'])+that.colorCellSize + that.colorCellMargin)
                    .attr('y', 0)
                    .attr('dy', that.cellAttrs['size']/2+that.horizonTextAttrs['font-size']/2)
                    .attr('text-anchor', that.horizonTextAttrs['text-anchor'])
                    .attr('font-size', that.horizonTextAttrs['font-size'])
                    .attr('font-weight', that.horizonTextAttrs['font-weight'])
                    .attr('font-family', that.horizonTextAttrs['font-family'])
                    .text((d) => d.name);

                const icony = that.cellAttrs['size']/2-that.horizonTextAttrs['font-size']/2+that.horizonTextAttrs['iconDy'];
                horizonTextinG.filter((d) => d.children.length>0)
                    .append('image')
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

                horizonTextinG.filter((d) => d.children.length>0)
                    .append('path')
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
                        const linelen = that.cellAttrs['size']*expandlen;
                        const x = that.horizonTextAttrs['font-size']/2;
                        return `M ${x} ${that.cellAttrs['size']} L ${x} ${that.cellAttrs['size']+linelen}`;
                    });

                const verticalTextinG = that.verticalTextinG.enter()
                    .append('g')
                    .attr('class', that.verticalTextAttrs['gClass'])
                    .attr('opacity', 0)
                    .attr('transform', (d, i) => `translate(${d.depth*that.verticalTextAttrs['leftMargin']}, 
                        ${i*that.cellAttrs['size']})`);

                verticalTextinG.transition()
                    .duration(that.createDuration)
                    .attr('opacity', 1)
                    .on('end', resolve);

                verticalTextinG.append('text')
                    .attr('x', (d) => (d.children.length===0?0:that.verticalTextAttrs['font-size'] +
                        that.verticalTextAttrs['iconMargin'])+that.colorCellSize + that.colorCellMargin)
                    .attr('y', 0)
                    .attr('dy', that.cellAttrs['size']/2+that.horizonTextAttrs['font-size']/2)
                    .attr('text-anchor', that.verticalTextAttrs['text-anchor'])
                    .attr('font-size', that.verticalTextAttrs['font-size'])
                    .attr('font-weight', that.verticalTextAttrs['font-weight'])
                    .attr('font-family', that.verticalTextAttrs['font-family'])
                    .text((d) => d.name);

                verticalTextinG.filter((d) => d.children.length>0)
                    .append('image')
                    .attr('xlink:href', (d) => '/static/images/'+(d.children.length>1?'arrow.png':'dot.png'))
                    .attr('x', 0)
                    .attr('y', icony)
                    .attr('width', that.verticalTextAttrs['font-size'])
                    .attr('height', that.verticalTextAttrs['font-size'])
                    .attr('transform', (d) => `rotate(${d.expand?90:0} 
                        ${that.verticalTextAttrs['font-size']/2} ${icony+that.verticalTextAttrs['font-size']/2})`)
                    .attr('cursor', 'pointer')
                    .on('click', function(e, d) {
                        if (d.children.length===1) return;
                        d.expand = !d.expand;
                        that.getDataAndRender();
                    });

                verticalTextinG.filter((d) => d.children.length>0)
                    .append('path')
                    .attr('stroke', that.verticalTextAttrs['indent-line-stroke'])
                    .attr('stroke-width', that.verticalTextAttrs['indent-line-stroke-width'])
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
                        const linelen = that.cellAttrs['size']*expandlen;
                        const x = that.verticalTextAttrs['font-size']/2;
                        return `M ${x} ${that.cellAttrs['size']} L ${x} ${that.cellAttrs['size']+linelen}`;
                    });

                const matrixCellsinG = that.matrixCellsinG.enter()
                    .append('g')
                    .attr('class', that.cellAttrs['gClass'])
                    .attr('opacity', 0)
                    .attr('cursor', (d)=>that.isHideCell(d)?'default':that.cellAttrs['cursor'])
                    .attr('transform', (d) => `translate(${d.column*that.cellAttrs['size']}, 
                        ${d.row*that.cellAttrs['size']})`)
                    .on('click', function(e, d) {
                        return;
                    })
                    .on('mouseover', function(e, d) {
                        if (that.isHideCell(d)) return;
                        const labelTarget = [];
                        const predictTarget = [];
                        for (const name of d.rowNode.leafs) {
                            labelTarget.push(that.name2index[name]);
                        }
                        for (const name of d.colNode.leafs) {
                            predictTarget.push(that.name2index[name]);
                        }
                        that.$emit('hoverConfusion', labelTarget, predictTarget);
                    })
                    .on('mouseout', function(e, d) {
                        that.$emit('hoverConfusion', undefined, undefined);
                    });

                matrixCellsinG.transition()
                    .duration(that.createDuration)
                    .attr('opacity', (d)=>(that.isHideCell(d)?0:1))
                    .on('end', resolve);

                matrixCellsinG.append('rect')
                    .attr('x', 0)
                    .attr('y', 0)
                    .attr('width', that.cellAttrs['size'])
                    .attr('height', that.cellAttrs['size'])
                    .attr('stroke', that.cellAttrs['stroke'])
                    .attr('stroke-width', that.cellAttrs['stroke-width'])
                    .attr('fill', (d)=>d.info.count===0?'rgb(255,255,255)':that.colorScale(d.info.val));


                // matrixCellsinG.filter((d) => d.info.val>0)
                //     .append('text')
                //     .attr('x', that.cellAttrs['size']/2)
                //     .attr('y', (that.cellAttrs['size']+that.cellAttrs['font-size'])/2)
                //     .attr('text-anchor', 'middle')
                //     .attr('font-size', that.cellAttrs['font-size'])
                //     .attr('font-weight', that.cellAttrs['font-weight'])
                //     .attr('font-family', that.cellAttrs['font-family'])
                //     .attr('opacity', 0)
                //     .attr('fill', that.cellAttrs['text-fill'])
                //     .text((d) => d.info.val);

                const directionScale = d3.scaleLinear([0, that.maxCellDirection], [0.4, 1]);

                matrixCellsinG.append('circle')
                    .attr('cx', that.cellAttrs['size']/2)
                    .attr('cy', that.cellAttrs['size']/2)
                    .attr('r', that.cellAttrs['size']/12)
                    .attr('fill', (d)=>d.info.count===0?'rgb(255,255,255)':that.cellAttrs['direction-color'])
                    .attr('opacity', (d)=>d.info.direction===undefined?0:d.info.direction[8]/Math.max(1, d3.max(d.info.direction)))
                    .attr('transform', (d)=>`translate(${that.cellAttrs['size']/2},${that.cellAttrs['size']/2})
                                             scale(${d.info.direction===undefined?0:Math.min(1, directionScale(d3.sum(d.info.direction)))})
                                             translate(${-that.cellAttrs['size']/2},${-that.cellAttrs['size']/2})`);

                for (let i = 0; i < 8; ++i) {
                    matrixCellsinG.filter((d) => d.info.count>0)
                        .append('polyline')
                        .attr('class', 'dir-'+i)
                        .attr('points', `${that.cellAttrs['size']/3},${that.cellAttrs['size']/2}
                                         ${that.cellAttrs['size']/18},${that.cellAttrs['size']/2}`)
                        .attr('fill', 'none')
                        .attr('stroke', 'currentColor')
                        .attr('marker-end', 'url(#arrow)')
                        .attr('opacity', (d)=>d.info.direction===undefined?0:d.info.direction[i]/Math.max(1, d3.max(d.info.direction)))
                        .attr('transform', (d)=>`translate(${that.cellAttrs['size']/2},${that.cellAttrs['size']/2})
                                                 scale(${d.info.direction===undefined?0:Math.min(1, directionScale(d3.sum(d.info.direction)))})
                                                 translate(${-that.cellAttrs['size']/2},${-that.cellAttrs['size']/2})
                                                 rotate(${i*45} ${that.cellAttrs['size']/2} ${that.cellAttrs['size']/2})`);
                }

                matrixCellsinG.append('path')
                    .attr('d', `M ${that.cellAttrs['size']*0.25} ${that.cellAttrs['size']*0.25} 
                        L ${that.cellAttrs['size']*0.75} ${that.cellAttrs['size']*0.75}`)
                    .attr('stroke', that.cellAttrs['slash-text-stroke'])
                    .attr('opacity', (d)=>d.info.count===0&&!that.showDirection?1:0);

                if (!that.showDirection) {
                    matrixCellsinG.select('circle')
                        .attr('fill-opacity', 0);
                } else {
                    matrixCellsinG.select('rect')
                        .attr('opacity', 0);
                }


                // show matrix legend text
                if (that.horizonLegend.attr('opacity')==0) {
                    that.horizonLegend
                        .transition()
                        .duration(that.createDuration)
                        .attr('opacity', 1)
                        .on('end', resolve);
                    that.verticalLegend
                        .transition()
                        .duration(that.createDuration)
                        .attr('opacity', 1)
                        .on('end', resolve);
                }


                if ((that.horizonTextinG.enter().size() === 0) && (that.verticalTextinG.enter().size() === 0) &&
                    (that.matrixCellsinG.enter().size() === 0)) {
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
                        ${i*that.cellAttrs['size']})`)
                    .on('end', resolve);

                const icony = that.cellAttrs['size']/2-that.horizonTextAttrs['font-size']/2+that.horizonTextAttrs['iconDy'];
                that.horizonTextinG.filter((d) => d.children.length>0)
                    .selectAll('image')
                    .attr('transform', (d) => `rotate(${d.expand?90:0} 
                        ${that.horizonTextAttrs['font-size']/2} ${icony+that.horizonTextAttrs['font-size']/2})`);

                that.horizonTextinG.filter((d) => d.children.length>0)
                    .selectAll('path')
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
                        const linelen = that.cellAttrs['size']*expandlen;
                        const x = that.horizonTextAttrs['font-size']/2;
                        return `M ${x} ${that.cellAttrs['size']} L ${x} ${that.cellAttrs['size']+linelen}`;
                    })
                    .on('end', resolve);

                that.horizonLegend
                    .transition()
                    .duration(that.updateDuration)
                    .attr('transform', `translate(-10,${that.matrixWidth/2}) rotate(270)`)
                    .on('end', resolve);

                that.verticalTextinG
                    .transition()
                    .duration(that.updateDuration)
                    .attr('transform', (d, i) => `translate(${d.depth*that.verticalTextAttrs['leftMargin']}, 
                        ${i*that.cellAttrs['size']})`)
                    .on('end', resolve);

                console.log(`translate(${that.maxHorizonTextWidth},${that.matrixWidth/2}) rotate(90)`);
                that.verticalLegend
                    .transition()
                    .duration(that.updateDuration)
                    .attr('transform', `translate(${that.maxHorizonTextWidth},${that.matrixWidth/2}) rotate(90)`)
                    .on('end', resolve);

                that.verticalTextinG.filter((d) => d.children.length>0)
                    .selectAll('image')
                    .attr('transform', (d) => `rotate(${d.expand?90:0} 
                        ${that.verticalTextAttrs['font-size']/2} ${icony+that.verticalTextAttrs['font-size']/2})`);

                that.verticalTextinG.filter((d) => d.children.length>0)
                    .selectAll('path')
                    .attr('stroke', that.verticalTextAttrs['indent-line-stroke'])
                    .attr('stroke-width', that.verticalTextAttrs['indent-line-stroke-width'])
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
                        const linelen = that.cellAttrs['size']*expandlen;
                        const x = that.verticalTextAttrs['font-size']/2;
                        return `M ${x} ${that.cellAttrs['size']} L ${x} ${that.cellAttrs['size']+linelen}`;
                    })
                    .on('end', resolve);

                that.matrixCellsinG
                    .transition()
                    .duration(that.updateDuration)
                    .attr('opacity', (d)=>(that.isHideCell(d)?0:1))
                    .attr('transform', (d) => `translate(${d.column*that.cellAttrs['size']}, 
                        ${d.row*that.cellAttrs['size']})`)
                    .on('end', resolve);

                that.matrixCellsinG.each(function(d) {
                    // eslint-disable-next-line no-invalid-this
                    d3.select(this).select('path')
                        .transition()
                        .duration(that.updateDuration)
                        .attr('opacity', (d)=>d.info.count===0&&!that.showDirection?1:0)
                        .on('end', resolve);
                });

                if (!that.showDirection) {
                    that.matrixCellsinG.each(function(d) {
                        // eslint-disable-next-line no-invalid-this
                        d3.select(this).select('rect')
                            .transition()
                            .duration(that.updateDuration)
                            .attr('fill', (d)=>d.info.count===0?'rgb(255,255,255)':that.colorScale(d.info.val))
                            .attr('opacity', 1)
                            .on('end', resolve);
                        // eslint-disable-next-line no-invalid-this
                        d3.select(this).select('circle')
                            .transition()
                            .duration(that.updateDuration)
                            .attr('fill-opacity', 0)
                            .on('end', resolve);
                        for (let i = 0; i < 8; ++i) {
                            // eslint-disable-next-line no-invalid-this
                            d3.select(this).select('.dir-'+i)
                                .transition()
                                .duration(that.updateDuration)
                                .attr('opacity', 0)
                                .on('end', resolve);
                        }
                    });
                } else {
                    const directionScale = d3.scaleLinear([0, that.maxCellDirection], [0.4, 1]);
                    that.matrixCellsinG.each(function(d) {
                        // eslint-disable-next-line no-invalid-this
                        d3.select(this).select('rect')
                            .transition()
                            .duration(that.updateDuration)
                            .attr('fill', (d)=>d.info.count===0?'rgb(255,255,255)':that.colorScale(d.info.val))
                            .attr('opacity', 0)
                            .on('end', resolve);
                        // eslint-disable-next-line no-invalid-this
                        d3.select(this).select('circle')
                            .transition()
                            .duration(that.updateDuration)
                            .attr('opacity', (d)=>d.info.direction[8]/Math.max(1, d3.max(d.info.direction)))
                            .attr('fill-opacity', 1)
                            .attr('transform', (d)=>`translate(${that.cellAttrs['size']/2},${that.cellAttrs['size']/2})
                                scale(${d.info.direction===undefined?0:Math.min(1, directionScale(d3.sum(d.info.direction)))})
                                translate(${-that.cellAttrs['size']/2},${-that.cellAttrs['size']/2})`)
                            .on('end', resolve);
                        for (let i = 0; i < 8; ++i) {
                            // eslint-disable-next-line no-invalid-this
                            d3.select(this).select('.dir-'+i)
                                .transition()
                                .duration(that.updateDuration)
                                .attr('opacity', (d)=>d.info.direction[i]/Math.max(1, d3.max(d.info.direction)))
                                .attr('transform', (d)=>`translate(${that.cellAttrs['size']/2},${that.cellAttrs['size']/2})
                                    scale(${d.info.direction===undefined?0:Math.min(1, directionScale(d3.sum(d.info.direction)))})
                                    translate(${-that.cellAttrs['size']/2},${-that.cellAttrs['size']/2})
                                    rotate(${i*45} ${that.cellAttrs['size']/2} ${that.cellAttrs['size']/2})`)
                                .on('end', resolve);
                        }
                    });
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

                that.verticalTextinG.exit()
                    .transition()
                    .duration(that.removeDuration)
                    .attr('opacity', 0)
                    .remove()
                    .on('end', resolve);

                that.matrixCellsinG.exit()
                    .transition()
                    .duration(that.removeDuration)
                    .attr('opacity', 0)
                    .remove()
                    .on('end', resolve);

                if ((that.horizonTextinG.exit().size() === 0) && (that.verticalTextinG.exit().size() === 0) &&
                    (that.matrixCellsinG.exit().size() === 0)) {
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
                const realSize = Math.min(svgRealWidth, svgRealHeight);
                let shiftx = 0;
                let shifty = 0;
                let scale = 1;
                if (that.svgWidth > realSize) {
                    scale = realSize/that.svgWidth/1.1;
                } else {
                    scale = 1;
                }
                console.log(that.svgWidth);
                shiftx = (svgRealWidth-scale*that.svgWidth)/2;
                shifty = (svgRealHeight-scale*that.svgWidth)/2;
                that.mainG.transition()
                    .duration(that.transformDuration)
                    .attr('transform', `translate(${shiftx} ${shifty}) scale(${scale})`)
                    .on('end', resolve);
                that.horizonTextG.transition()
                    .duration(that.transformDuration)
                    .attr('transform', `translate(${that.leftCornerSize-that.maxHorizonTextWidth}, ${that.leftCornerSize+that.textMatrixMargin})`)
                    .on('end', resolve);
                that.verticalTextG.transition()
                    .duration(that.transformDuration)
                    .attr('transform', `translate(${that.leftCornerSize+that.textMatrixMargin}, ${that.leftCornerSize}) rotate(-90)`)
                    .on('end', resolve);
                that.matrixCellsG.transition()
                    .duration(that.transformDuration)
                    .attr('transform', `translate(${that.leftCornerSize+that.textMatrixMargin}, ${that.leftCornerSize+that.textMatrixMargin})`)
                    .on('end', resolve);
                that.legendG.transition()
                    .duration(that.transformDuration)
                    .attr('transform', `translate(${that.leftCornerSize+that.textMatrixMargin},
                        ${that.leftCornerSize+that.textMatrixMargin+that.matrixWidth+5})`)
                    .on('end', resolve);
            });
        },
        getTwoCellConfusion: function(nodea, nodeb) {
            const infoMap = {
                'count': 0,
                'val': 0,
            };
            if (this.showDirection) {
                infoMap['direction'] = [];
                for (let i = 0; i < 9; ++i) infoMap['direction'].push(0);
            }
            for (const leafa of nodea.leafs) {
                for (const leafb of nodeb.leafs) {
                    infoMap.count += this.baseMatrix[0][this.name2index[leafa]][this.name2index[leafb]];
                    if (this.returnMode !== 'count') {
                        infoMap.val += this.baseMatrix[1][this.name2index[leafa]][this.name2index[leafb]]*
                                       this.baseMatrix[0][this.name2index[leafa]][this.name2index[leafb]];
                    }
                    if (this.showDirection) {
                        for (let i = 0; i < 9; ++i) {
                            infoMap.direction[i] += this.baseMatrix[this.baseMatrix.length-1][this.name2index[leafa]][this.name2index[leafb]][i];
                        }
                    }
                }
            }
            if (this.returnMode === 'count') infoMap.val = infoMap.count;
            else infoMap.val /= infoMap.count;
            return infoMap;
        },
        drawLegend: function() {
            const that = this;
            // https://observablehq.com/@d3/color-legend
            const drawLegend = function({
                color,
                title,
                tickSize = 6,
                width = 320,
                height = 44 + tickSize,
                marginTop = 18,
                marginRight = 0,
                marginBottom = 16 + tickSize,
                marginLeft = 0,
                ticks = width / 64,
                tickFormat,
                tickValues,
            } = {}) {
                const ramp = function(color, n = 256) {
                    const canvas = that.drawLegend.canvas || (that.drawLegend.canvas = document.createElement('canvas'));
                    canvas.width = n;
                    canvas.height = 1;
                    const context = canvas.getContext('2d');
                    for (let i = 0; i < n; ++i) {
                        context.fillStyle = color(i / (n - 1));
                        context.fillRect(i, 0, 1, 1);
                    }
                    return canvas;
                };
                const tickAdjust = (g) => g.selectAll('.tick line').attr('y1', marginTop + marginBottom - height);
                let x;

                // Continuous
                if (color.interpolator) {
                    x = Object.assign(color.copy()
                        .interpolator(d3.interpolateRound(marginLeft, width - marginRight)),
                    {range() {
                        return [marginLeft, width - marginRight];
                    }});

                    that.legendG.append('image')
                        .attr('x', marginLeft)
                        .attr('y', marginTop)
                        .attr('width', width - marginLeft - marginRight)
                        .attr('height', height - marginTop - marginBottom)
                        .attr('preserveAspectRatio', 'none')
                        .attr('xlink:href', ramp(color.interpolator()).toDataURL());

                    // scaleSequentialQuantile doesn’t implement ticks or tickFormat.
                    if (!x.ticks) {
                        if (tickValues === undefined) {
                            const n = Math.round(ticks + 1);
                            tickValues = d3.range(n).map((i) => d3.quantile(color.domain(), i / (n - 1)));
                        }
                        if (typeof tickFormat !== 'function') {
                            tickFormat = d3.format(tickFormat === undefined ? ',f' : tickFormat);
                        }
                    }
                }
                that.legendG.append('g')
                    .attr('transform', `translate(0,${height - marginBottom})`)
                    .call(d3.axisBottom(x)
                        .ticks(ticks, typeof tickFormat === 'string' ? tickFormat : undefined)
                        .tickFormat(typeof tickFormat === 'function' ? tickFormat : undefined)
                        .tickSize(tickSize)
                        .tickValues(tickValues))
                    .call(tickAdjust)
                    .call((g) => g.select('.domain').remove())
                    .call((g) => g.append('text')
                        .attr('x', marginLeft)
                        .attr('y', marginTop + marginBottom - height - 6)
                        .attr('fill', 'currentColor')
                        .attr('text-anchor', 'start')
                        .attr('font-weight', 'bold')
                        .attr('class', 'title')
                        .text(title));

                return that.legendG.node();
            };
            this.legendG.selectAll('*').remove();
            drawLegend(
                {
                    color: this.colorScale,
                    title: this.returnMode,
                    width: this.legendWidth,
                    ticks: 6,
                },
            );
        },
        isHideCell: function(cell) {
            const isHideNode = function(node) {
                return node.expand===true && node.children.length>0;
            };
            return isHideNode(this.showNodes[cell.row]) || isHideNode(this.showNodes[cell.column]);
        },
    },
};
</script>
