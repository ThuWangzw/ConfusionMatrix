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
            <g id="class-statistics-g" transform="translate(0, 0)"></g>
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
        showMode: {
            type: String,
            default: 'normal',
        },
        classStatistics: {
            type: Array,
            default: undefined,
        },
        normalizationMode: {
            type: String,
            default: 'total',
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
            return this.svg.select('g#horizon-text-g');
        },
        verticalTextG: function() {
            return this.svg.select('g#vertical-text-g');
        },
        matrixCellsG: function() {
            return this.svg.select('g#matrix-cells-g');
        },
        legendG: function() {
            return this.svg.select('g#legend-g');
        },
        mainG: function() {
            return this.svg.select('g#main-g');
        },
        statG: function() {
            return this.svg.select('g#class-statistics-g');
        },
        horizonLegend: function() {
            return this.svg.select('text#horizon-legend');
        },
        verticalLegend: function() {
            return this.svg.select('text#vertical-legend');
        },
        maxHorizonTextWidth: function() {
            let maxwidth = 0;
            for (const node of this.showNodes) {
                const textwidth = this.getTextWidth(node.name,
                    `${this.horizonTextAttrs['font-weight']} ${this.horizonTextAttrs['font-size']}px ${this.horizonTextAttrs['font-family']}`);
                const arrowIconNum = node.children.length===0?node.depth:node.depth+1;
                maxwidth = Math.max(maxwidth, this.horizonTextAttrs['leftMargin']*node.depth + textwidth +
                    arrowIconNum*(this.horizonTextAttrs['font-size'] + this.horizonTextAttrs['iconMargin'])+
                    this.colorCellSize+this.colorCellMargin+15);
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
            if (this.returnMode==='avg_iou' || this.returnMode==='avg_acc') {
                return d3.scaleSequential([1, 0], ['rgb(226, 232, 224)', 'rgb(56, 99, 140)']).clamp(true);
            } else if (this.returnMode==='avg_label_size' || this.returnMode==='avg_predict_size') {
                return d3.scaleSequential([0, 1], ['rgb(226, 232, 224)', 'rgb(56, 99, 140)']).clamp(true);
            } else {
                if (this.normalizationMode !== 'total' && this.returnMode === 'count') {
                    return d3.scaleSequential([0, 1], ['rgb(226, 232, 224)', 'rgb(56, 99, 140)']).clamp(true);
                } else return d3.scaleSequential([0, this.maxCellValue], ['rgb(226, 232, 224)', 'rgb(56, 99, 140)']).clamp(true);
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
            this.legendExist = false;
            this.getDataAndRender();
        },
        showMode: function() {
            this.getDataAndRender();
        },
        normalizationMode: function() {
            if (this.returnMode !== 'count') return;
            this.legendExist = false;
            this.getDataAndRender();
        },
        classStatistics: function() {
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
            classStatShow: [],
            // layout elements
            horizonTextinG: null,
            verticalTextinG: null,
            matrixCellsinG: null,
            classStatinG: null,
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
                'size-color': ['rgb(227,227,227)', 'rgb(255,102,0)', 'rgb(95,198,181)'],
            },
            statAttrs: {
                'gClass': 'class-stat-g',
                'width': 50,
                'height': 5,
                'font-family': 'Comic Sans MS',
                'font-weight': 'normal',
                'font-size': 13,
                'bg-color': 'rgb(227,227,227)',
                'color': 'steelblue',
            },
            legendExist: false,
            // buffer
            maxCellCount: 0,
            maxCellValue: 0,
            maxCellDirection: 0,
            columnSum: [],
            rowSum: [],
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
            if (this.confusionMatrix===undefined || this.labelHierarchy===undefined || this.classStatistics === undefined) {
                return;
            }
            // get nodes to show
            this.showNodes = this.getShowNodes(this.hierarchy);
            // get cells to render
            this.cells = [];
            this.classStatShow = [];
            this.classStatShow.push({
                val: d3.mean(this.classStatistics),
                row: -1.5,
                key: 'all',
            });
            this.maxCellValue = 0;
            this.maxCellDirection = 0;
            this.maxCellCount = 0;
            this.columnSum = [];
            this.rowSum = [];
            for (let i=0; i<this.showNodes.length; i++) {
                this.columnSum.push(0);
                this.rowSum.push(0);
            }
            for (let i=0; i<this.showNodes.length; i++) {
                const nodea = this.showNodes[i];
                if (i < this.showNodes.length - 1) {
                    let tmp = 0;
                    for (const leaf of nodea.leafs) {
                        tmp += this.classStatistics[this.name2index[leaf]];
                    }
                    this.classStatShow.push({
                        val: tmp / nodea.leafs.length,
                        row: i,
                        key: nodea.name,
                    });
                }
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
                    this.rowSum[i] += cell.info.val;
                    this.columnSum[j] += cell.info.val;
                    if (i === this.showNodes.length-1 || j === this.showNodes.length-1) {
                        cell.info.direction = undefined;
                        cell.info.sizeCmp = undefined;
                        if (i === this.showNodes.length-1 && j === this.showNodes.length-1) cell.info.sizeDist = undefined;
                    } else cell.info.sizeDist = undefined;
                    this.cells.push(cell);
                    if (!this.isHideCell(cell)) {
                        this.maxCellValue = Math.max(this.maxCellValue, cell.info.val);
                    }
                    if (!this.isHideCell(cell) && (i!=j || this.returnMode!=='count') && i!==this.showNodes.length-1 && j!==this.showNodes.length-1) {
                        this.maxCellCount = Math.max(this.maxCellCount, cell.info.count);
                        if (i!=j && cell.info.direction!==undefined) {
                            this.maxCellDirection = Math.max(this.maxCellDirection, d3.sum(cell.info.direction));
                        }
                    }
                }
            }
            this.render();
        },
        render: async function() {
            this.horizonTextinG = this.horizonTextG.selectAll('g.'+this.horizonTextAttrs['gClass']).data(this.showNodes, (d)=>d.name);
            this.verticalTextinG = this.verticalTextG.selectAll('g.'+this.verticalTextAttrs['gClass']).data(this.showNodes, (d)=>d.name);
            this.matrixCellsinG = this.matrixCellsG.selectAll('g.'+this.cellAttrs['gClass']).data(this.cells, (d)=>d.key);
            this.classStatinG = this.statG.selectAll('g.'+this.statAttrs['gClass']).data(this.classStatShow, (d)=>d.key);
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
                        that.legendExist = false;
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
                        that.legendExist = false;
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
                        that.$emit('clickCell', d);
                    })
                    .on('mouseover', function(e, d) {
                        if (that.isHideCell(d)) return;
                        // eslint-disable-next-line no-invalid-this
                        const cell = d3.select(this);
                        cell.select('rect').attr('stroke-width', '3px');
                        const labelTarget = [];
                        const predictTarget = [];
                        for (const name of d.rowNode.leafs) {
                            labelTarget.push(that.name2index[name]);
                        }
                        for (const name of d.colNode.leafs) {
                            predictTarget.push(that.name2index[name]);
                        }
                        that.$emit('hoverConfusion', labelTarget, predictTarget);
                        if (that.showMode === 'direction' && d.info.direction !== undefined) {
                            // eslint-disable-next-line no-invalid-this
                            d3.select(this).select('.dir-circle')
                                .transition()
                                .duration(that.updateDuration)
                                .attr('transform', '');
                            for (let i = 0; i < 8; ++i) {
                                // eslint-disable-next-line no-invalid-this
                                d3.select(this).select('.dir-'+i)
                                    .transition()
                                    .duration(that.updateDuration)
                                    .attr('transform', `rotate(${i*45} ${that.cellAttrs['size']/2} ${that.cellAttrs['size']/2})`);
                            }
                        } else if (that.showMode === 'sizeComparison' && d.info.sizeCmp !== undefined) {
                            const radius = that.cellAttrs['size']/3;
                            for (let i = 0; i < 3; ++i) {
                                // eslint-disable-next-line no-invalid-this
                                d3.select(this).select(`#size-circle-${i}`)
                                    .transition()
                                    .duration(that.updateDuration)
                                    .attrTween('d', (d) => d.info.count===0?
                                        d3.arc().innerRadius(0).outerRadius(0).startAngle(0).endAngle(0):
                                        d3.arc().innerRadius(0).outerRadius(radius)
                                            .startAngle(d.info.sizeCmpAngle[i]).endAngle(d.info.sizeCmpAngle[i+1]));
                            }
                        }
                    })
                    .on('mouseout', function(e, d) {
                        that.$emit('hoverConfusion', undefined, undefined);
                        // eslint-disable-next-line no-invalid-this
                        const cell = d3.select(this);
                        cell.select('rect').attr('stroke-width', that.cellAttrs['stroke-width']);
                        if (that.showMode === 'direction' && d.info.direction !== undefined) {
                            const directionScale = d3.scaleLinear([0, that.maxCellDirection], [0.4, 1]);
                            // eslint-disable-next-line no-invalid-this
                            d3.select(this).select('.dir-circle')
                                .transition()
                                .duration(that.updateDuration)
                                .attr('transform', (d)=>`translate(${that.cellAttrs['size']/2},${that.cellAttrs['size']/2})
                                    scale(${d.info.direction===undefined?0:Math.min(1, directionScale(d3.sum(d.info.direction)))})
                                    translate(${-that.cellAttrs['size']/2},${-that.cellAttrs['size']/2})`);
                            for (let i = 0; i < 8; ++i) {
                                // eslint-disable-next-line no-invalid-this
                                d3.select(this).select('.dir-'+i)
                                    .transition()
                                    .duration(that.updateDuration)
                                    .attr('transform', (d)=>`translate(${that.cellAttrs['size']/2},${that.cellAttrs['size']/2})
                                        scale(${d.info.direction===undefined?0:Math.min(1, directionScale(d3.sum(d.info.direction)))})
                                        translate(${-that.cellAttrs['size']/2},${-that.cellAttrs['size']/2})
                                        rotate(${i*45} ${that.cellAttrs['size']/2} ${that.cellAttrs['size']/2})`);
                            }
                        } else if (that.showMode === 'sizeComparison' && d.info.sizeCmp !== undefined) {
                            const sizeScale = d3.scaleLinear([0, Math.sqrt(that.maxCellCount)], [0.2, 1]);
                            const radius = that.cellAttrs['size']/3;
                            for (let i = 0; i < 3; ++i) {
                                // eslint-disable-next-line no-invalid-this
                                d3.select(this).select(`#size-circle-${i}`)
                                    .transition()
                                    .duration(that.updateDuration)
                                    .attrTween('d', (d) => d.info.count===0?
                                        d3.arc().innerRadius(0).outerRadius(0).startAngle(0).endAngle(0):
                                        d3.arc().innerRadius(0).outerRadius(Math.min(1, sizeScale(Math.sqrt(d.info.count)))*radius)
                                            .startAngle(d.info.sizeCmpAngle[i]).endAngle(d.info.sizeCmpAngle[i+1]));
                            }
                        }
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
                    .attr('fill', (d)=>d.info.count===0?'rgb(255,255,255)':that.getFillColor(d));

                for (let i = 0; i < 5; ++i) {
                    matrixCellsinG.filter((d) => d.info.sizeDist!==undefined)
                        .append('rect')
                        .attr('class', 'dist')
                        .attr('id', 'distRect-' + i)
                        .attr('x', 2.5+5*i)
                        .attr('width', 5)
                        .attr('height', (d) => that.getDistHeight(d.info.sizeDist[i]))
                        .attr('y', (d) => that.cellAttrs.size - 1 - that.getDistHeight(d.info.sizeDist[i]))
                        .attr('fill', 'rgb(150,150,150)');
                    matrixCellsinG.filter((d) => d.info.sizeDist!==undefined)
                        .append('polyline')
                        .attr('class', 'dist')
                        .attr('id', 'distPolyline-' + i + '-0')
                        .attr('stroke-width', (d) => d.info.sizeDist[i]>=1000?1:0)
                        .attr('stroke', 'rgb(75,75,75)')
                        .attr('fill', 'rgb(255,255,255)')
                        .attr('points', `${3+5*i},6 ${5+5*i},4.5 ${7+5*i},6`);
                    matrixCellsinG.filter((d) => d.info.sizeDist!==undefined)
                        .append('polyline')
                        .attr('class', 'dist')
                        .attr('id', 'distPolyline-' + i + '-1')
                        .attr('stroke-width', (d) => d.info.sizeDist[i]>=4000?1:0)
                        .attr('stroke', 'rgb(75,75,75)')
                        .attr('fill', 'rgb(255,255,255)')
                        .attr('points', `${3+5*i},4 ${5+5*i},2.5 ${7+5*i},4`);
                    matrixCellsinG.filter((d) => d.info.sizeDist!==undefined)
                        .append('polyline')
                        .attr('class', 'dist')
                        .attr('id', 'distPolyline-' + i + '-2')
                        .attr('stroke-width', (d) => d.info.sizeDist[i]>=10000?1:0)
                        .attr('stroke', 'rgb(75,75,75)')
                        .attr('fill', 'rgb(255,255,255)')
                        .attr('points', `${3+5*i},2 ${5+5*i},0.5 ${7+5*i},2`);
                }
                matrixCellsinG.filter((d) => d.info.sizeDist!==undefined)
                    .append('line')
                    .attr('class', 'dist')
                    .attr('stroke', 'rgb(75,75,75)')
                    .attr('x1', 2)
                    .attr('x2', 28)
                    .attr('y1', 29.5)
                    .attr('y2', 29.5);

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

                // direction mode: circle and arrows
                const directionScale = d3.scaleLinear([0, that.maxCellDirection], [0.4, 1]);

                matrixCellsinG.append('circle')
                    .attr('class', 'dir-circle')
                    .attr('cx', that.cellAttrs['size']/2)
                    .attr('cy', that.cellAttrs['size']/2)
                    .attr('r', (d)=>d.info.direction===undefined?0:
                        that.cellAttrs['size']*5/36*d.info.direction[8]/Math.max(1, d3.max(d.info.direction)))
                    .attr('fill', 'currentColor')
                    .attr('transform', (d)=>`translate(${that.cellAttrs['size']/2},${that.cellAttrs['size']/2})
                                             scale(${d.info.direction===undefined?0:Math.min(1, directionScale(d3.sum(d.info.direction)))})
                                             translate(${-that.cellAttrs['size']/2},${-that.cellAttrs['size']/2})`);

                for (let i = 0; i < 8; ++i) {
                    matrixCellsinG.filter((d) => d.info.count>0)
                        .append('polyline')
                        .attr('class', 'dir-'+i)
                        .attr('points', (d)=>`${that.cellAttrs['size']/3},${that.cellAttrs['size']/2}
                                         ${d.info.direction===undefined?
        that.cellAttrs['size']/18:
        that.cellAttrs['size']*Math.min(2/9, 1/3-5/18*d.info.direction[i]/Math.max(1, d3.max(d.info.direction)))},
                                         ${that.cellAttrs['size']/2}`)
                        .attr('fill', 'none')
                        .attr('stroke', 'currentColor')
                        .attr('marker-end', 'url(#arrow)')
                        .attr('opacity', (d)=>d.info.count===0||d.info.direction===undefined||d.info.direction[i]===0?0:1)
                        .attr('transform', (d)=>`translate(${that.cellAttrs['size']/2},${that.cellAttrs['size']/2})
                                                 scale(${d.info.direction===undefined?0:Math.min(1, directionScale(d3.sum(d.info.direction)))})
                                                 translate(${-that.cellAttrs['size']/2},${-that.cellAttrs['size']/2})
                                                 rotate(${i*45} ${that.cellAttrs['size']/2} ${that.cellAttrs['size']/2})`);
                }

                // sizeComparison mode: two circles
                // matrixCellsinG.filter((d) => d.info.count>0&&d.info.sizeCmp!==undefined).append('circle')
                //     .attr('class', 'size-circle')
                //     .attr('id', 'size-large-circle')
                //     .attr('cx', that.cellAttrs['size']/2)
                //     .attr('cy', that.cellAttrs['size']/2)
                //     .attr('r', that.cellAttrs['size']/3)
                //     .attr('fill', (d)=>Math.abs(d.info.sizeCmp[0]-d.info.sizeCmp[1])<10?
                //         'rgb(237,237,237)':d.info.sizeCmp[0]>d.info.sizeCmp[1]?'rgb(243,158,112)':'rgb(95,198,181)');

                // matrixCellsinG.filter((d) => d.info.count>0&&d.info.sizeCmp!==undefined).append('circle')
                //     .attr('class', 'size-circle')
                //     .attr('id', 'size-small-circle')
                //     .attr('cx', that.cellAttrs['size']/2)
                //     .attr('cy', that.cellAttrs['size']/2)
                //     .attr('r', that.cellAttrs['size']/6)
                //     .attr('fill', (d)=>Math.abs(d.info.sizeCmp[0]-d.info.sizeCmp[1])<10?
                //         'rgb(227,227,227)':d.info.sizeCmp[0]<d.info.sizeCmp[1]?'rgb(243,158,112)':'rgb(95,198,181)');

                // sizeComparison mode: piecharts
                const sizeScale = d3.scaleLinear([0, Math.sqrt(that.maxCellCount)], [0.2, 1]);
                const radius = that.cellAttrs['size']/3;
                for (let i = 0; i < 3; ++i) {
                    matrixCellsinG.filter((d) => d.info.count>0&&d.info.sizeCmp!==undefined).append('path')
                        .attr('class', 'size-circle')
                        .attr('id', `size-circle-${i}`)
                        .attr('transform', `translate(${that.cellAttrs.size/2} ${that.cellAttrs.size/2})`)
                        .attr('fill', that.cellAttrs['size-color'][i])
                        .attr('d', (d) => d3.arc().innerRadius(0).outerRadius(Math.min(1, sizeScale(Math.sqrt(d.info.count)))*radius)
                            .startAngle(d.info.sizeCmpAngle[i]).endAngle(d.info.sizeCmpAngle[i+1])());
                }

                // normal mode: sign for empty cells
                matrixCellsinG.append('path')
                    .attr('id', 'empty-line')
                    .attr('d', `M ${that.cellAttrs['size']*0.25} ${that.cellAttrs['size']*0.25} 
                        L ${that.cellAttrs['size']*0.75} ${that.cellAttrs['size']*0.75}`)
                    .attr('stroke', that.cellAttrs['slash-text-stroke'])
                    .attr('opacity', (d)=>d.info.count===0&&that.showMode==='normal'?1:0);

                matrixCellsinG.append('title')
                    .text((d)=>`count: ${d.info.count}`);

                if (that.showMode!=='direction') {
                    matrixCellsinG.select('.dir-circle')
                        .attr('r', 0);
                }
                if (that.showMode!=='normal') {
                    matrixCellsinG.select('rect')
                        .attr('opacity', 0);
                }
                if (that.showMode!=='sizeComparison') {
                    matrixCellsinG.selectAll('.size-circle')
                        .attr('opacity', 0);
                    matrixCellsinG.selectAll('.dist')
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

                const classStatinG = that.classStatinG.enter()
                    .append('g')
                    .attr('opacity', 0)
                    .attr('class', that.statAttrs['gClass']);
                classStatinG.transition()
                    .duration(that.createDuration)
                    .attr('opacity', 1)
                    .on('end', resolve);
                classStatinG.append('rect')
                    .attr('class', 'statBgRect')
                    .attr('x', 0)
                    .attr('y', (d)=>that.cellAttrs.size*d.row+that.cellAttrs.size-that.statAttrs.height)
                    .attr('width', that.statAttrs.width)
                    .attr('height', that.statAttrs.height)
                    .attr('fill', that.statAttrs['bg-color']);
                classStatinG.append('rect')
                    .attr('class', 'statRect')
                    .attr('x', 0)
                    .attr('y', (d)=>that.cellAttrs.size*d.row+that.cellAttrs.size-that.statAttrs.height)
                    .attr('width', (d)=>that.statAttrs.width*d.val)
                    .attr('height', that.statAttrs.height)
                    .attr('fill', that.statAttrs['color']);
                classStatinG.append('text')
                    .text((d)=>d.val.toFixed(3))
                    .attr('y', (d)=>that.cellAttrs.size*d.row+that.cellAttrs.size-that.statAttrs.height-3)
                    .attr('x', 0)
                    .attr('font-size', that.statAttrs['font-size'])
                    .attr('font-weight', that.statAttrs['font-weight'])
                    .attr('font-family', that.statAttrs['font-family']);


                if ((that.horizonTextinG.enter().size() === 0) && (that.verticalTextinG.enter().size() === 0) &&
                    (that.matrixCellsinG.enter().size() === 0) && (that.classStatinG.enter().size() === 0)) {
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
                    d3.select(this).select('#empty-line')
                        .transition()
                        .duration(that.updateDuration)
                        .attr('opacity', (d)=>d.info.count===0&&that.showMode==='normal'?1:0)
                        .on('end', resolve);
                    // eslint-disable-next-line no-invalid-this
                    d3.select(this).select('title')
                        .text((d)=>`count: ${d.info.count}`);
                });

                if (that.showMode!=='direction') {
                    that.matrixCellsinG.each(function(d) {
                        // eslint-disable-next-line no-invalid-this
                        d3.select(this).select('.dir-circle')
                            .transition()
                            .duration(that.updateDuration)
                            .attr('r', 0)
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
                }
                if (that.showMode!=='normal') {
                    that.matrixCellsinG.each(function(d) {
                        // eslint-disable-next-line no-invalid-this
                        d3.select(this).select('rect')
                            .transition()
                            .duration(that.updateDuration)
                            .attr('fill', (d)=>d.info.count===0?'rgb(255,255,255)':that.getFillColor(d))
                            .attr('opacity', 0)
                            .on('end', resolve);
                    });
                }
                if (that.showMode!=='sizeComparison') {
                    that.matrixCellsinG.each(function(d) {
                        // eslint-disable-next-line no-invalid-this
                        d3.select(this).selectAll('.size-circle')
                            .transition()
                            .duration(that.updateDuration)
                            .attr('opacity', 0)
                            .on('end', resolve);
                        // eslint-disable-next-line no-invalid-this
                        d3.select(this).selectAll('.dist')
                            .transition()
                            .duration(that.updateDuration)
                            .attr('opacity', 0)
                            .on('end', resolve);
                    });
                }

                if (that.showMode==='normal') {
                    that.matrixCellsinG.each(function(d) {
                        // eslint-disable-next-line no-invalid-this
                        d3.select(this).select('rect')
                            .transition()
                            .duration(that.updateDuration)
                            .attr('fill', (d)=>d.info.count===0?'rgb(255,255,255)':that.getFillColor(d))
                            .attr('opacity', 1)
                            .on('end', resolve);
                    });
                } else if (that.showMode==='direction') {
                    const directionScale = d3.scaleLinear([0, that.maxCellDirection], [0.4, 1]);
                    that.matrixCellsinG.each(function(d) {
                        // eslint-disable-next-line no-invalid-this
                        d3.select(this).select('.dir-circle')
                            .transition()
                            .duration(that.updateDuration)
                            .attr('r', (d)=>d.info.direction===undefined?0:
                                that.cellAttrs['size']*5/36*d.info.direction[8]/Math.max(1, d3.max(d.info.direction)))
                            .attr('transform', (d)=>`translate(${that.cellAttrs['size']/2},${that.cellAttrs['size']/2})
                                scale(${d.info.direction===undefined?0:Math.min(1, directionScale(d3.sum(d.info.direction)))})
                                translate(${-that.cellAttrs['size']/2},${-that.cellAttrs['size']/2})`)
                            .on('end', resolve);
                        for (let i = 0; i < 8; ++i) {
                            // eslint-disable-next-line no-invalid-this
                            d3.select(this).select('.dir-'+i)
                                .transition()
                                .duration(that.updateDuration)
                                .attr('points', (d)=>`${that.cellAttrs['size']/3},${that.cellAttrs['size']/2}
                                                ${d.info.direction===undefined?that.cellAttrs['size']/18:
        that.cellAttrs['size']*Math.min(2/9, 1/3-5/18*d.info.direction[i]/Math.max(1, d3.max(d.info.direction)))},
                                                ${that.cellAttrs['size']/2}`)
                                .attr('opacity', (d)=>d.info.count===0||d.info.direction===undefined||d.info.direction[i]===0?0:1)
                                .attr('transform', (d)=>`translate(${that.cellAttrs['size']/2},${that.cellAttrs['size']/2})
                                    scale(${d.info.direction===undefined?0:Math.min(1, directionScale(d3.sum(d.info.direction)))})
                                    translate(${-that.cellAttrs['size']/2},${-that.cellAttrs['size']/2})
                                    rotate(${i*45} ${that.cellAttrs['size']/2} ${that.cellAttrs['size']/2})`)
                                .on('end', resolve);
                        }
                    });
                } else if (that.showMode==='sizeComparison') {
                    const sizeScale = d3.scaleLinear([0, Math.sqrt(that.maxCellCount)], [0.2, 1]);
                    const radius = that.cellAttrs['size']/3;
                    that.matrixCellsinG.each(function(d) {
                        // eslint-disable-next-line no-invalid-this
                        // d3.select(this).select('#size-small-circle')
                        //     .transition()
                        //     .duration(that.updateDuration)
                        //     .attr('fill', (d)=>Math.abs(d.info.sizeCmp[0]-d.info.sizeCmp[1])<10?
                        //         'rgb(227,227,227)':d.info.sizeCmp[0]<d.info.sizeCmp[1]?'rgb(243,158,112)':'rgb(95,198,181)')
                        //     .on('end', resolve);
                        // eslint-disable-next-line no-invalid-this
                        // d3.select(this).select('#size-large-circle')
                        //     .transition()
                        //     .duration(that.updateDuration)
                        //     .attr('fill', (d)=>Math.abs(d.info.sizeCmp[0]-d.info.sizeCmp[1])<10?
                        //         'rgb(237,237,237)':d.info.sizeCmp[0]>d.info.sizeCmp[1]?'rgb(243,158,112)':'rgb(95,198,181)')
                        //     .on('end', resolve);
                        // eslint-disable-next-line no-invalid-this
                        // d3.select(this).selectAll('.size-circle')
                        //     .transition()
                        //     .duration(that.updateDuration)
                        //     .attr('opacity', (d)=>d.info.count===0?0:1)
                        //     .on('end', resolve);

                        // sizeComparison mode: piecharts
                        for (let i = 0; i < 3; ++i) {
                            // eslint-disable-next-line no-invalid-this
                            d3.select(this).select(`#size-circle-${i}`)
                                .transition()
                                .duration(that.updateDuration)
                                .attrTween('d', (d) => d.info.count===0?
                                    d3.arc().innerRadius(0).outerRadius(0).startAngle(0).endAngle(0):
                                    d3.arc().innerRadius(0).outerRadius(Math.min(1, sizeScale(Math.sqrt(d.info.count)))*radius)
                                        .startAngle(d.info.sizeCmpAngle[i]).endAngle(d.info.sizeCmpAngle[i+1]))
                                .attr('opacity', (d)=>d.info.count===0?0:1)
                                .on('end', resolve);
                        }
                        for (let i = 0; i < 5; ++i) {
                            // eslint-disable-next-line no-invalid-this
                            d3.select(this).select(`#distRect-${i}`)
                                .transition()
                                .duration(that.updateDuration)
                                .attr('opacity', 1)
                                .attr('height', (d) => that.getDistHeight(d.info.sizeDist[i]))
                                .attr('y', (d) => that.cellAttrs.size - 1 - that.getDistHeight(d.info.sizeDist[i]))
                                .on('end', resolve);
                            // eslint-disable-next-line no-invalid-this
                            d3.select(this).select(`#distPolyline-${i}-0`)
                                .transition()
                                .duration(that.updateDuration)
                                .attr('opacity', (d) => d.info.sizeDist[i]>=1000?1:0)
                                .attr('stroke-width', (d) => d.info.sizeDist[i]>=1000?1:0)
                                .on('end', resolve);
                            // eslint-disable-next-line no-invalid-this
                            d3.select(this).select(`#distPolyline-${i}-1`)
                                .transition()
                                .duration(that.updateDuration)
                                .attr('opacity', (d) => d.info.sizeDist[i]>=4000?1:0)
                                .attr('stroke-width', (d) => d.info.sizeDist[i]>=4000?1:0)
                                .on('end', resolve);
                            // eslint-disable-next-line no-invalid-this
                            d3.select(this).select(`#distPolyline-${i}-2`)
                                .transition()
                                .duration(that.updateDuration)
                                .attr('opacity', (d) => d.info.sizeDist[i]>=10000?1:0)
                                .attr('stroke-width', (d) => d.info.sizeDist[i]>=10000?1:0)
                                .on('end', resolve);
                            // eslint-disable-next-line no-invalid-this
                            d3.select(this).select('line')
                                .transition()
                                .duration(that.updateDuration)
                                .attr('opacity', 1)
                                .on('end', resolve);
                        }
                    });
                }
                that.classStatinG.each(function(d) {
                    // eslint-disable-next-line no-invalid-this
                    d3.select(this).select('.statBgRect')
                        .transition()
                        .duration(that.updateDuration)
                        .attr('y', (d)=>that.cellAttrs.size*d.row+that.cellAttrs.size-that.statAttrs.height)
                        .on('end', resolve);
                    // eslint-disable-next-line no-invalid-this
                    d3.select(this).select('.statRect')
                        .transition()
                        .duration(that.updateDuration)
                        .attr('width', (d)=>that.statAttrs.width*d.val)
                        .attr('y', (d)=>that.cellAttrs.size*d.row+that.cellAttrs.size-that.statAttrs.height)
                        .on('end', resolve);
                    // eslint-disable-next-line no-invalid-this
                    d3.select(this).select('text')
                        .transition()
                        .duration(that.updateDuration)
                        .text((d)=>d.val.toFixed(3))
                        .attr('y', (d)=>that.cellAttrs.size*d.row+that.cellAttrs.size-that.statAttrs.height-3)
                        .on('end', resolve);
                });
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

                that.classStatinG.exit()
                    .transition()
                    .duration(that.removeDuration)
                    .attr('opacity', 0)
                    .remove()
                    .on('end', resolve);

                if ((that.horizonTextinG.exit().size() === 0) && (that.verticalTextinG.exit().size() === 0) &&
                    (that.matrixCellsinG.exit().size() === 0) && (that.classStatinG.exit().size() === 0)) {
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
                if (that.svgWidth*1.1 > realSize) {
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
                that.statG.transition()
                    .duration(that.transformDuration)
                    .attr('transform', `translate(${that.leftCornerSize+that.textMatrixMargin+that.matrixWidth+20}, 
                        ${that.leftCornerSize+that.textMatrixMargin})`)
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
                'sizeCmp': [0, 0, 0],
                'sizeDist': [0, 0, 0, 0, 0],
            };
            if (this.showMode==='direction') {
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
                    if (this.showMode==='direction') {
                        for (let i = 0; i < 9; ++i) {
                            infoMap.direction[i] += this.baseMatrix[this.baseMatrix.length-1][this.name2index[leafa]][this.name2index[leafb]][i];
                        }
                    }
                    if (leafa === 'background' || leafb === 'background') {
                        for (let i = 0; i < 5 &&
                            i < this.baseMatrix[this.baseMatrix.length-2][this.name2index[leafa]][this.name2index[leafb]].length; ++i) {
                            infoMap.sizeDist[i] += this.baseMatrix[this.baseMatrix.length-2][this.name2index[leafa]][this.name2index[leafb]][i];
                        }
                    } else {
                        infoMap.sizeCmp[1] += this.baseMatrix[this.baseMatrix.length-2][this.name2index[leafa]][this.name2index[leafb]][0];
                        infoMap.sizeCmp[2] += this.baseMatrix[this.baseMatrix.length-2][this.name2index[leafa]][this.name2index[leafb]][1];
                    }
                }
            }
            infoMap.sizeCmp[0] = infoMap.count - infoMap.sizeCmp[1] - infoMap.sizeCmp[2];
            if (infoMap.count > 0) {
                infoMap.sizeCmpAngle = [0];
                for (let i = 0; i < 3; ++i) {
                    infoMap.sizeCmpAngle.push(infoMap.sizeCmpAngle[i] + infoMap.sizeCmp[i]/infoMap.count * 2 * Math.PI);
                }
            }
            if (this.returnMode === 'count') infoMap.val = infoMap.count;
            else if (infoMap.count > 0) infoMap.val /= infoMap.count;
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
        getDistHeight: function(y) {
            if (y===0) return 0;
            return d3.scaleLinear([0, 1000], [2, this.cellAttrs.size-8])(Math.min(y, 1000));
        },
        getFillColor: function(d) {
            if (this.normalizationMode === 'total' || this.normalizationMode === 'none') return this.colorScale(d.info.val);
            else if (this.normalizationMode === 'row') return this.colorScale(d.info.val / this.rowSum[d.row]);
            else return this.colorScale(d.info.val / this.columnSum[d.column]);
        },
    },
};
</script>
