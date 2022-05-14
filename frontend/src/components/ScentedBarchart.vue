<template>
    <svg :id="widgetId" width="100%" height="100%" ref="svg">
        <g id="all-data-g"></g>
        <g id="select-data-g"></g>
    </svg>
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
    },
    data: function() {
        return {
            globalAttrs: {
                'marginTop': 20, // top margin, in pixels
                'marginRight': 10, // right margin, in pixels
                'marginBottom': 30, // bottom margin, in pixels
                'marginLeft': 30, // left margin, in pixels
                'width': 300, // outer width of chart, in pixels
                'height': 100, // outer height of chart, in pixels
                'insetLeft': 0.5, // inset left edge of bar
                'insetRight': 0.5, // inset right edge of bar:
                'xType': d3.scaleLinear, // type of x-scale
                'yType': d3.scaleLinear, // type of y-scale
                'unselectFill': 'rgb(237,237,237)',
            },
            textAttrs: {
                'font-family': 'Comic Sans MS',
                'font-weight': 'bold',
                'font-size': 12,
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
        render: async function() {
            const xRange = [this.globalAttrs['marginLeft'], this.globalAttrs['width'] - this.globalAttrs['marginRight']]; // [left, right]
            const yRange = [this.globalAttrs['height'] - this.globalAttrs['marginBottom'], this.globalAttrs['marginTop']]; // [bottom, top]
            const xDomain = [0, 1];
            const yDomain = [0, Math.log10(d3.max(this.allData))+1];
            this.xScale = this.globalAttrs['xType'](xDomain, xRange);
            this.yScale = this.globalAttrs['yType'](yDomain, yRange);
            const that = this;
            if (this.drawAxis === false) {
                this.drawAxis = true;
                const xFormat = undefined;
                let yFormat = undefined;
                const xAxis = d3.axisBottom(this.xScale).ticks(this.globalAttrs['width'] / 80, xFormat).tickSizeOuter(0);
                const yAxis = d3.axisLeft(this.yScale).ticks(Math.floor(this.globalAttrs['height'] / 40), yFormat);
                yFormat = this.yScale.tickFormat(100, yFormat);

                this.mainSvg
                    .append('g')
                    .attr('transform', `translate(${this.globalAttrs['marginLeft']},0)`)
                    .call(yAxis)
                    .call((g) => g.select('.domain').remove())
                    .call((g) => g.selectAll('.tick line').clone()
                        .attr('x2', this.globalAttrs['width'] - this.globalAttrs['marginLeft'] - this.globalAttrs['marginRight'])
                        .attr('stroke-opacity', 0.1))
                    .call((g) => g.append('text')
                        .attr('x', -this.globalAttrs['marginLeft'])
                        .attr('y', 10)
                        .attr('fill', 'currentColor')
                        .attr('text-anchor', 'start')
                        .attr('font-family', that.textAttrs['font-family'])
                        .attr('font-weight', that.textAttrs['font-weight'])
                        .attr('font-size', that.textAttrs['font-size'])
                        .text('log count'));

                this.mainSvg
                    .append('g')
                    .attr('transform', `translate(0,${this.globalAttrs['height'] - this.globalAttrs['marginBottom']})`)
                    .call(xAxis)
                    .call((g) => g.append('text')
                        .attr('x', this.globalAttrs['width'] - this.globalAttrs['marginRight'])
                        .attr('y', 27)
                        .attr('fill', 'currentColor')
                        .attr('text-anchor', 'end')
                        .attr('font-family', that.textAttrs['font-family'])
                        .attr('font-weight', that.textAttrs['font-weight'])
                        .attr('font-size', that.textAttrs['font-size'])
                        .text(this.title));

                this.mainSvg
                    .append('text')
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
                        that.selectDataG.call(that.brush.move, null);
                    });

                this.selectDataG
                    .call(this.brush.extent([[this.globalAttrs['marginLeft'], this.globalAttrs['marginTop']],
                        [this.globalAttrs['width'] - this.globalAttrs['marginRight'], this.globalAttrs['height'] - this.globalAttrs['marginBottom']]])
                        .on('end', function({selection}) {
                            const len = that.globalAttrs['width'] - that.globalAttrs['marginRight']-that.globalAttrs['marginLeft'];
                            let x1 = 0;
                            let x2 = 1;
                            if (selection!==null) {
                                x1 = Math.floor((selection[0] - that.globalAttrs['marginLeft'])/len*10)/10;
                                x2 = Math.ceil((selection[1] - that.globalAttrs['marginLeft'])/len*10)/10-(1e-5);
                            }
                            const query = {};
                            query[that.queryKey] = [x1, x2];
                            that.$emit('hoverBarchart', query);
                        }));
            }
            const selectDataBins = [];
            const allDataBins = [];
            for (let i = 0; i < this.allData.length; ++i) {
                selectDataBins.push({
                    'val': this.selectData[i],
                    'x0': i*0.1,
                    'x1': (i+1)*0.1,
                });
                allDataBins.push({
                    'val': this.allData[i],
                    'x0': i*0.1,
                    'x1': (i+1)*0.1,
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
                    .attr('y', (d, i) => that.yScale(Math.log10(Math.max(1, d.val))))
                    .attr('height', (d, i) => that.yScale(0) - that.yScale(Math.log10(Math.max(1, d.val))))
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
                    .attr('y', (d, i) => that.yScale(Math.log10(Math.max(1, d.val))))
                    .attr('height', (d, i) => that.yScale(0) - that.yScale(Math.log10(Math.max(1, d.val))))
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
                        .attr('height', that.yScale(0) - that.yScale(Math.log10(Math.max(1, d.val))))
                        .on('end', resolve);
                });
                that.selectDataRectG.each(function(d, i) {
                    // eslint-disable-next-line no-invalid-this
                    d3.select(this).select('rect')
                        .transition()
                        .duration(that.updateDuration)
                        .attr('y', that.yScale(Math.log10(Math.max(1, d.val))))
                        .attr('height', that.yScale(0) - that.yScale(Math.log10(Math.max(1, d.val))))
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
    },
    mounted: function() {
        this.globalAttrs['width'] = this.$refs.svg.clientWidth;
        this.globalAttrs['height'] = this.$refs.svg.clientHeight;
    },
};
</script>
