<template>
    <div id="data-content">
        <div id="confusion-matrix-container">
            <confusion-matrix ref="matrix" id="confusion-matrix" :showColor="true">
            </confusion-matrix>
        </div>
    </div>
</template>

<script>
import ConfusionMatrix from './ConfusionMatrix.vue';

export default {
    components: {ConfusionMatrix},
    name: 'DataView',
    data() {
        return {
        };
    },
    methods: {
    },
};
</script>

<style scoped>
#data-content {
    width: 100%;
    height: 100%;
    display: flex;
}

#confusion-matrix-container {
  width: 100%;
  height: 100%;
  display: flex;
  margin: 10px 10px 10px 10px;
}
</style>
