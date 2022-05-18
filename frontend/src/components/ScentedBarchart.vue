<template>
    <svg :id="widgetId" width="100%" height="100%" ref="svg"></svg>
</template>

<script>
import * as d3 from 'd3';
window.d3 = d3;
import Util from './Util.vue';
import GlobalVar from './GlovalVar.vue';
import {brushX} from 'd3-brush';

export default {
    name: 'ScentedBarchart',
    mixins: [Util, GlobalVar],
    props: {
        allData: {
            type: Array,
            default: undefined,
        },
        title: {
            type: String,
            default: '',
        },
        selectData: {
            type: Array,
            default: undefined,
        },
        queryKey: {
            type: String,
            default: '',
        },
        xSplit: {
            type: Array,
            default: undefined,
        },
        displayMode: {
            type: String,
            default: 'log',
        },
        barNum: {
            type: Number,
            default: 10,
        },
    },
    computed: {
        widgetId: function() {
            return 'scented-barchart-svg-'+this.title;
        },
        mainSvg: function() {
            return d3.select('#'+this.widgetId);
        },
        alldataG: function() {
            return this.mainSvg.select('#all-data-g');
        },
        selectDataG: function() {
            return this.mainSvg.select('#select-data-g');
        },
    },
    mounted: function() {

    },
    watch: {
        allData: function() {
            this.render();
        },
        selectData: function() {
            this.render();
        },
        displayMode: function() {
            this.render();
        },
    },
    data: function() {
        return {
            globalAttrs: {
                'marginTop': 10, // top margin, in pixels
                'marginRight': 0, // right margin, in pixels
                'marginBottom': 0, // bottom margin, in pixels
                'marginLeft': 60, // left margin, in pixels
                'width': 300, // outer width of chart, in pixels
                'height': 100, // outer height of chart, in pixels
                'insetLeft': 0.3, // inset left edge of bar
                'insetRight': 0.3, // inset right edge of bar:
                'xType': d3.scaleLinear, // type of x-scale
                'yType': d3.scaleLinear, // type of y-scale
                'unselectFill': 'rgb(237,237,237)',
            },
            textAttrs: {
                'font-family': 'Comic Sans MS',
                'font-weight': 'bold',
                'font-size': 10,
            },
            selectDataRectG: null,
            allDataRectG: null,
            drawAxis: false,
            xScale: undefined,
            yScale: undefined,
            brush: brushX(),
        };
    },
    methods: {
        createResetBrush: function() {
            const that = this;
            this.mainSvg
                .append('text')
                .attr('id', 'remove-brush-button')
                .attr('x', this.globalAttrs['width'] - this.globalAttrs['marginRight'])
                .attr('y', 10)
                .attr('fill', 'currentColor')
                .attr('text-anchor', 'end')
                .attr('cursor', 'pointer')
                .attr('font-family', that.textAttrs['font-family'])
                .attr('font-weight', that.textAttrs['font-weight'])
                .attr('font-size', that.textAttrs['font-size'])
                .text('reset brush')
                .on('click', ()=> {
                    that.mainSvg.call(that.brush.move, null);
                    that.mainSvg.selectAll('#remove-brush-button').remove();
                });
        },
        render: async function() {
            const xRange = [this.globalAttrs['marginLeft'], this.globalAttrs['width'] - this.globalAttrs['marginRight']]; // [left, right]
            const yRange = [this.globalAttrs['height'] - this.globalAttrs['marginBottom'], this.globalAttrs['marginTop']]; // [bottom, top]
            const xDomain = [-0.05, 1.05];
            const yDomain = [0, this.cal(d3.max(this.allData))];
            this.xScale = this.globalAttrs['xType'](xDomain, xRange);
            this.yScale = this.globalAttrs['yType'](yDomain, yRange);
            const that = this;
            if (this.drawAxis === false) {
                this.drawAxis = true;

                this.mainSvg
                    .append('line')
                    .attr('transform', `translate(${this.globalAttrs['marginLeft']},${this.yScale(0)})`)
                    .attr('stroke-opacity', 0.2)
                    .attr('stroke', 'currentColor')
                    .attr('x2', this.globalAttrs['width'] - this.globalAttrs['marginLeft'] - this.globalAttrs['marginRight']);

                this.mainSvg
                    .append('g')
                    .attr('id', 'all-data-g');

                this.mainSvg
                    .append('g')
                    .attr('id', 'select-data-g');

                this.mainSvg
                    .call(this.brush.extent([[this.xScale(0), this.globalAttrs['marginTop']],
                        [this.xScale(1), this.globalAttrs['height'] - this.globalAttrs['marginBottom']]])
                        .on('end', function({selection}) {
                            that.createResetBrush();
                            const len = that.xScale(1) - that.xScale(0);
                            let x1 = that.xSplit[0];
                            let x2 = that.xSplit[that.xSplit.length-1];
                            if (selection!==null) {
                                x1 = that.xSplit[Math.floor((selection[0] - that.xScale(0))/len*that.barNum)]+(1e-5);
                                x2 = that.xSplit[Math.ceil((selection[1] - that.xScale(0))/len*that.barNum)];
                            }
                            const query = {};
                            query[that.queryKey] = [x1, x2];
                            that.$emit('hoverBarchart', query);
                        }));

                this.mainSvg
                    .append('text')
                    .attr('x', 20)
                    .attr('y', this.globalAttrs['height']-3)
                    .attr('fill', 'currentColor')
                    .attr('font-family', that.textAttrs['font-family'])
                    .attr('font-weight', that.textAttrs['font-weight'])
                    .attr('font-size', that.textAttrs['font-size'])
                    .text(this.title);

                that.mainSvg
                    .append('circle')
                    .attr('cx', 13)
                    .attr('cy', this.globalAttrs['height']-6)
                    .attr('r', 2)
                    .attr('fill', 'currentColor');
            }
            const selectDataBins = [];
            const allDataBins = [];
            const rectWidth = 1 / this.barNum;
            for (let i = 0; i < this.allData.length; ++i) {
                selectDataBins.push({
                    'val': this.selectData[i],
                    'x0': i * rectWidth,
                    'x1': (i+1) * rectWidth,
                });
                allDataBins.push({
                    'val': this.allData[i],
                    'x0': i * rectWidth,
                    'x1': (i+1) * rectWidth,
                });
            }
            this.allDataRectG = this.alldataG.selectAll('g.allDataRect').data(allDataBins);
            this.selectDataRectG = this.selectDataG.selectAll('g.selectDataRect').data(selectDataBins);
            await this.remove();
            await this.update();
            await this.transform();
            await this.create();
        },
        create: async function() {
            const that = this;
            return new Promise((resolve, reject) => {
                const allDataRectG = that.allDataRectG.enter()
                    .append('g')
                    .attr('class', 'allDataRect');

                allDataRectG.transition()
                    .duration(that.createDuration)
                    .attr('opacity', 1)
                    .on('end', resolve);

                allDataRectG.append('rect')
                    .attr('x', (d) => that.xScale(d.x0) + that.globalAttrs['insetLeft'])
                    .attr('width', (d) => Math.max(0, that.xScale(d.x1) - that.xScale(d.x0) -
                                                      that.globalAttrs['insetLeft'] - that.globalAttrs['insetRight']))
                    .attr('y', (d, i) => that.yScale(that.cal(d.val)))
                    .attr('height', (d, i) => that.yScale(0) - that.yScale(that.cal(d.val)))
                    .attr('fill', that.globalAttrs['unselectFill'])
                    .append('title')
                    .text((d, i) => [`${d.x0.toFixed(1)} ≤ x < ${d.x1.toFixed(1)}`, `quantity: ${d.val}`].join('\n'));

                const selectDataRectG = that.selectDataRectG.enter()
                    .append('g')
                    .attr('class', 'selectDataRect');

                selectDataRectG.transition()
                    .duration(that.createDuration)
                    .attr('opacity', 1)
                    .on('end', resolve);

                selectDataRectG.append('rect')
                    .attr('x', (d) => that.xScale(d.x0) + that.globalAttrs['insetLeft'])
                    .attr('width', (d) => Math.max(0, that.xScale(d.x1) - that.xScale(d.x0) -
                                                      that.globalAttrs['insetLeft'] - that.globalAttrs['insetRight']))
                    .attr('y', (d, i) => that.yScale(that.cal(d.val)))
                    .attr('height', (d, i) => that.yScale(0) - that.yScale(that.cal(d.val)))
                    .attr('fill', 'steelblue');

                if ((that.selectDataRectG.enter().size() === 0) && (that.allDataRectG.enter().size() === 0)) {
                    resolve();
                }
            });
        },
        update: async function() {
            const that = this;
            return new Promise((resolve, reject) => {
                that.allDataRectG.each(function(d, i) {
                    // eslint-disable-next-line no-invalid-this
                    d3.select(this).select('rect')
                        .transition()
                        .duration(that.updateDuration)
                        .attr('y', that.yScale(that.cal(d.val)))
                        .attr('height', that.yScale(0) - that.yScale(that.cal(d.val)))
                        .on('end', resolve);
                });
                that.selectDataRectG.each(function(d, i) {
                    // eslint-disable-next-line no-invalid-this
                    d3.select(this).select('rect')
                        .transition()
                        .duration(that.updateDuration)
                        .attr('y', that.yScale(that.cal(d.val)))
                        .attr('height', that.yScale(0) - that.yScale(that.cal(d.val)))
                        .on('end', resolve);
                });

                if ((that.selectDataRectG.size() === 0) && (that.allDataRectG.size() === 0)) {
                    resolve();
                }
            });
        },
        remove: async function() {
            const that = this;
            return new Promise((resolve, reject) => {
                that.allDataRectG.exit()
                    .transition()
                    .duration(that.removeDuration)
                    .attr('opacity', 0)
                    .remove()
                    .on('end', resolve);
                that.selectDataRectG.exit()
                    .transition()
                    .duration(that.removeDuration)
                    .attr('opacity', 0)
                    .remove()
                    .on('end', resolve);

                if ((that.selectDataRectG.exit().size() === 0) && (that.allDataRectG.exit().size() === 0)) {
                    resolve();
                }
            });
        },
        transform: async function() {
        },
        cal: function(d) {
            if (this.displayMode === 'log') return Math.log10(Math.max(1, d));
            else if (this.displayMode === 'linear') return d;
        },
    },
    mounted: function() {
        this.globalAttrs['width'] = this.$refs.svg.clientWidth;
        this.globalAttrs['height'] = this.$refs.svg.clientHeight;
    },
};
</script>
