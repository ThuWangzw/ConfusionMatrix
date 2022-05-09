<template>
    <svg id="numerical-svg" width="100%" height="100%" ref="svg">
        <g id="main-g" transform="translate(0,0)">
            <g id="legend-g" :transform="`translate(5,${leftCornerSize/2-25})`"></g>
            <g id="horizon-text-g" :transform="`translate(${leftCornerSize-maxHorizonTextWidth}, ${leftCornerSize+textMatrixMargin})`"></g>
            <g id="vertical-text-g" :transform="`translate(${leftCornerSize+textMatrixMargin}, ${leftCornerSize}) rotate(-90)`"></g>
            <g id="matrix-cells-g" :transform="`translate(${leftCornerSize+textMatrixMargin}, ${leftCornerSize+textMatrixMargin})`"></g>
        </g>
    </svg>
</template>

<script>
import Util from './Util.vue';
import GlobalVar from './GlovalVar.vue';
import PriorityQueue from 'priorityqueue';

export default {
    mixins: [Util, GlobalVar],
    props: {
        numericalMatrix: {
            type: Object,
            default: undefined,
        },
        numericalMatrixType: {
            type: String,
            default: 'size',
        },
    },
    watch: {
        numericalMatrix: function() {
            this.getDataAndRender();
        },
    },
    computed: {
        partitions: function() {
            if (this.numericalMatrix===undefined || this.numericalMatrix.partitions===undefined) {
                return [];
            } else {
                return this.numericalMatrix.partitions;
            }
        },
        rawMatrix: function() {
            if (this.numericalMatrix===undefined || this.numericalMatrix.matrix===undefined) {
                return [];
            } else {
                return this.numericalMatrix.matrix;
            }
        },
        leftCornerSize: function() {
            return this.legendWidth;
        },
        matrixWidth: function() {
            return (this.partitions.length-1) * this.cellAttrs['size'];
        },
        maxHorizonTextWidth: function() {
            let maxwidth = 0;
            for (const partition of this.partitions) {
                const textwidth = this.getTextWidth(partition,
                    `${this.horizonTextAttrs['font-weight']} ${this.horizonTextAttrs['font-size']}px ${this.horizonTextAttrs['font-family']}`);
                maxwidth = Math.max(maxwidth, textwidth);
            }
            return maxwidth;
        },
        svgWidth: function() {
            return this.leftCornerSize+this.textMatrixMargin+this.matrixWidth;
        },
        colorScale: function() {
            return d3.scaleSequential([0, this.submaxCellValue], ['rgb(255, 255, 255)', 'rgb(8, 48, 107)']).clamp(true);
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
        mainG: function() {
            return d3.selectAll('g#main-g');
        },
        legendG: function() {
            return d3.select('g#legend-g');
        },
        legendWidth: function() {
            return Math.max(200, this.maxHorizonTextWidth);
        },
    },
    data: function() {
        return {
            textMatrixMargin: 10,
            cells: [], // main layout data
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
            },
            submaxCellValue: 0,
        };
    },
    methods: {
        getDataAndRender: function() {
            this.cells = [];

            const pq = new PriorityQueue({
                comparator: function(a, b) {
                    return a > b ? 1 : a < b ? -1 : 0;
                },
            });
            for (let i=0; i<this.rawMatrix.length; i++) {
                for (let j=0; j<this.rawMatrix[i].length; j++) {
                    this.cells.push({
                        key: this.partitions[i]+','+this.partitions[j],
                        val: this.rawMatrix[i][j],
                        col: j,
                        row: i,
                    });
                    pq.push(this.rawMatrix[i][j]);
                }
            }
            for (let i=0; i<2 && pq.length>0; i++) {
                pq.pop();
            }
            if (pq.length>0) {
                this.submaxCellValue = pq.top();
            } else {
                this.submaxCellValue = 0;
            }

            this.render();
        },
        render: async function() {
            this.horizonTextinG = this.horizonTextG.selectAll('g.'+this.horizonTextAttrs['gClass']).data(this.partitions, (d)=>d);
            this.verticalTextinG = this.verticalTextG.selectAll('g.'+this.verticalTextAttrs['gClass']).data(this.partitions, (d)=>d);
            this.matrixCellsinG = this.matrixCellsG.selectAll('g.'+this.cellAttrs['gClass']).data(this.cells, (d)=>d.key);

            this.drawLegend();
            await this.remove();
            await this.update();
            await this.transform();
            await this.create();
        },
        create: async function() {
            const that = this;
            return new Promise((resolve, reject) => {
                const horizonTextinG = that.horizonTextinG.enter()
                    .append('g')
                    .attr('class', that.horizonTextAttrs['gClass'])
                    .attr('opacity', 0)
                    .attr('transform', (d, i) => `translate(0, ${i*that.cellAttrs['size']})`);

                horizonTextinG.transition()
                    .duration(that.createDuration)
                    .attr('opacity', 1)
                    .on('end', resolve);

                horizonTextinG.append('text')
                    .attr('x', 0)
                    .attr('y', 0)
                    .attr('dy', that.horizonTextAttrs['font-size']/2)
                    .attr('text-anchor', that.horizonTextAttrs['text-anchor'])
                    .attr('font-size', that.horizonTextAttrs['font-size'])
                    .attr('font-weight', that.horizonTextAttrs['font-weight'])
                    .attr('font-family', that.horizonTextAttrs['font-family'])
                    .text((d) => d);

                const verticalTextinG = that.verticalTextinG.enter()
                    .append('g')
                    .attr('class', that.verticalTextAttrs['gClass'])
                    .attr('opacity', 0)
                    .attr('transform', (d, i) => `translate(0,${i*that.cellAttrs['size']})`);

                verticalTextinG.transition()
                    .duration(that.createDuration)
                    .attr('opacity', 1)
                    .on('end', resolve);

                verticalTextinG.append('text')
                    .attr('x', 0)
                    .attr('y', 0)
                    .attr('dy', that.verticalTextAttrs['font-size']/2)
                    .attr('text-anchor', that.verticalTextAttrs['text-anchor'])
                    .attr('font-size', that.verticalTextAttrs['font-size'])
                    .attr('font-weight', that.verticalTextAttrs['font-weight'])
                    .attr('font-family', that.verticalTextAttrs['font-family'])
                    .text((d) => d);

                const matrixCellsinG = that.matrixCellsinG.enter()
                    .append('g')
                    .attr('class', that.cellAttrs['gClass'])
                    .attr('opacity', 0)
                    .attr('cursor', that.cellAttrs['cursor'])
                    .attr('transform', (d) => `translate(${d.col*that.cellAttrs['size']}, 
                        ${d.row*that.cellAttrs['size']})`);

                matrixCellsinG.transition()
                    .duration(that.createDuration)
                    .attr('opacity', 1)
                    .on('end', resolve);

                matrixCellsinG.append('rect')
                    .attr('x', 0)
                    .attr('y', 0)
                    .attr('width', that.cellAttrs['size'])
                    .attr('height', that.cellAttrs['size'])
                    .attr('stroke', that.cellAttrs['stroke'])
                    .attr('stroke-width', that.cellAttrs['stroke-width'])
                    .attr('fill', (d)=>that.colorScale(d.val));

                matrixCellsinG.filter((d) => d.val===0)
                    .append('path')
                    .attr('d', `M ${that.cellAttrs['size']*0.25} ${that.cellAttrs['size']*0.25} 
                        L ${that.cellAttrs['size']*0.75} ${that.cellAttrs['size']*0.75}`)
                    .attr('stroke', that.cellAttrs['slash-text-stroke']);

                matrixCellsinG.filter((d) => d.val>0)
                    .append('text')
                    .attr('x', that.cellAttrs['size']/2)
                    .attr('y', (that.cellAttrs['size']+that.cellAttrs['font-size'])/2)
                    .attr('text-anchor', 'middle')
                    .attr('font-size', that.cellAttrs['font-size'])
                    .attr('font-weight', that.cellAttrs['font-weight'])
                    .attr('font-family', that.cellAttrs['font-family'])
                    .attr('opacity', 0)
                    .attr('fill', that.cellAttrs['text-fill'])
                    .text((d) => d.val);


                if ((that.horizonTextinG.enter().size() === 0) && (that.verticalTextinG.enter().size() === 0) &&
                    (that.matrixCellsinG.enter().size() === 0)) {
                    resolve();
                }
            });
        },
        update: async function() {

        },
        remove: async function() {

        },
        transform: async function() {
            const that = this;
            return new Promise((resolve, reject) => {
                // compute transform
                const svgRealWidth = that.$refs.svg.clientWidth;
                const svgRealHeight = that.$refs.svg.clientHeight;
                const realSize = Math.min(svgRealWidth, svgRealHeight);
                let shiftx = 0;
                let shifty = 0;
                let scale = 1;
                if (that.svgWidth > realSize) {
                    scale = realSize/that.svgWidth;
                } else {
                    scale = 1;
                }
                shiftx = (svgRealWidth-scale*that.svgWidth)/2;
                shifty = (svgRealHeight-scale*that.svgWidth)/2;
                that.mainG.transition()
                    .duration(that.transformDuration)
                    .attr('transform', `translate(${shiftx} ${shifty}) scale(${scale})`)
                    .on('end', resolve);
            });
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
                    title: 'Counts',
                    width: this.legendWidth,
                    ticks: 5,
                },
            );
        },
    },
    mounted: function() {
        this.getDataAndRender();
    },
};
</script>
